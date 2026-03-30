import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# 导入你的模型和数据集
from model import PST_UNet
from datasets import SARDataset

def setup_logger(log_dir):
    """配置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger('TrainLogger')
    logger.setLevel(logging.INFO)
    
    # 防止重复添加 handler
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 文件输出
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        
        # 控制台输出
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

def main():
    parser = argparse.ArgumentParser(description="PST-UNet 训练脚本")
    parser.add_argument('--data_dir', type=str, default=r'G:\VSCODE-G\PST_Dataset', help='数据集根目录')
    parser.add_argument('--domain', type=str, default='DB', choices=['DB', 'Linear'], help='数据域')
    parser.add_argument('--epochs', type=int, default=100, help='训练总轮数')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--max_val', type=float, default=255.0, help='反量化除数 (uint8为255, uint16为65535)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='权重保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志保存目录')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader的进程数')
    args = parser.parse_args()

    # 1. 设置日志和目录
    logger = setup_logger(args.log_dir)
    logger.info("=" * 50)
    logger.info("🚀 开始 PST-UNet 训练任务")
    logger.info(f"参数配置: {vars(args)}")
    os.makedirs(args.save_dir, exist_ok=True)

    # 2. 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"💻 运行设备: {device}")

    # 3. 准备数据
    logger.info("📦 加载数据集中...")
    try:
        train_dataset = SARDataset(base_dir=args.data_dir, domain=args.domain, mode='train', max_val=args.max_val)
        val_dataset = SARDataset(base_dir=args.data_dir, domain=args.domain, mode='test', max_val=args.max_val)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        logger.info(f"✅ 数据加载成功! 训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")
    except Exception as e:
        logger.error(f"❌ 数据加载失败: {e}")
        return

    # 4. 初始化模型
    model = PST_UNet(in_channels=2, out_channels=1, base_dim=64)
    if torch.cuda.device_count() > 1:
        logger.info(f"🔥 检测到 {torch.cuda.device_count()} 张显卡，启动多卡并行模式！")
        model = nn.DataParallel(model)
    model = model.to(device)

    # 5. 损失函数与优化器
    criterion = nn.L1Loss() # 要求使用的 L1 Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') # 用于混合精度训练

    # 6. 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # -------------------- 训练阶段 --------------------
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
        
        for inputs, targets in train_bar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix({'L1Loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_dataset)
        
        # -------------------- 验证阶段 --------------------
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for inputs, targets in val_bar:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                val_bar.set_postfix({'L1Loss': f"{loss.item():.4f}"})
                
        val_loss /= len(val_dataset)
        
        # 日志记录当前 epoch 结果
        logger.info(f"Epoch [{epoch:03d}/{args.epochs:03d}] - Train L1 Loss: {train_loss:.6f} | Val L1 Loss: {val_loss:.6f}")
        
        # -------------------- 保存最优模型 --------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            
            # 如果使用了 DataParallel，保存时去掉 module. 前缀
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, save_path)
            
            logger.info(f"🌟 新的最优模型已保存! 验证集 L1 Loss 降至 {best_val_loss:.6f} -> {save_path}")

    logger.info("🎉 训练任务全部完成!")
    logger.info(f"🏆 最终最优验证集 L1 Loss: {best_val_loss:.6f}")
    logger.info("=" * 50)

if __name__ == '__main__':
    main()
