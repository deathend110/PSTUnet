import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SARDataset(Dataset):
    def __init__(self, base_dir, domain='DB', mode='train', max_val=255.0):
        """
        SAR 数据集加载器
        :param base_dir: 数据集根目录，例如 'Sequence_Dataset_mixrate_1-1_1.8'
        :param domain: 'DB' 或 'Linear'
        :param mode: 'train' 或 'test'
        :param max_val: 反量化除数 (如果是 uint16 保存的请填 65535.0，如果是 uint8 则填 255.0)
        """
        self.domain = domain
        self.max_val = max_val
        
        # 映射 mode 到你的底层文件夹名字
        folder_mode = 'traindata' if mode == 'train' else 'testdata'
        
        # 组装绝对/相对路径
        self.data_dir = os.path.join(base_dir, domain, folder_mode)
        
        # 搜索该文件夹下的所有 .mat 文件 (也就是你用 h5write 保存的 HDF5 文件)
        self.file_paths = sorted(glob.glob(os.path.join(self.data_dir, '*.mat')))
        
        if len(self.file_paths) == 0:
            print(f"⚠️ 警告: 在 {self.data_dir} 下没有找到任何 .mat 文件！请检查路径。")
            
        # 根据你传入的 domain，智能匹配 HDF5 内部的 key 名字
        if self.domain == 'DB':
            self.key_in = '/seq_input'
            self.key_gt = '/seq_GT'
        elif self.domain == 'Linear':
            self.key_in = '/seq_input_L'
            self.key_gt = '/seq_GT_L'
        else:
            raise ValueError("domain 参数必须是 'DB' 或 'Linear'")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # 使用 context manager 确保多进程读取时的线程安全
        with h5py.File(file_path, 'r') as f:
            # 1. 读取 16 位整数数据，并直接转为 float32
            # 注意：h5py 读出来的形状是 (T, W, H)，即 (16, 512, 512)
            seq_in = np.array(f[self.key_in]).astype(np.float32)
            seq_gt = np.array(f[self.key_gt]).astype(np.float32)
            
            # 读取帧类型 (16 帧)，强制拍平为一维向量 (16,)
            frame_type = np.array(f['/frame_type']).astype(np.float32).flatten()
            
        # 2. 轴转置 (Crucial Step!) 
        # 将 (T, W, H) 转置为 (T, H, W)，匹配 PyTorch 的标准张量方向
        seq_in = seq_in.transpose(0, 2, 1)
        seq_gt = seq_gt.transpose(0, 2, 1)
        
        # 3. 反量化恢复到 0.0 ~ 1.0 的物理真实值域
        seq_in = seq_in / self.max_val
        seq_gt = seq_gt / self.max_val
        
        # 4. 构建 Channel 1: 帧类型扩展掩码 (Mask)
        # 获取 T, H, W (16, 512, 512)
        T, H, W = seq_in.shape
        # 生成一个基础底板
        mask = np.ones((T, H, W), dtype=np.float32)
        # 利用广播机制，把一维的 frame_type(16,) 扩充到三维并相乘
        mask = mask * frame_type[:, np.newaxis, np.newaxis]
        
        # 5. 组装输入张量 (拼接到通道维) -> 形状: (2, 16, 512, 512)
        # 通道 0: 原始退化图像
        # 通道 1: 帧类型的 Mask
        input_tensor = np.stack([seq_in, mask], axis=0)
        
        # 6. 为 GT 增加通道维度 -> 形状: (1, 16, 512, 512)
        # 因为模型输出通道为 1，所以目标也要是 1 通道
        gt_tensor = seq_gt[np.newaxis, :, :, :]
        
        return torch.from_numpy(input_tensor), torch.from_numpy(gt_tensor)

# ==========================================
# 本地测试脚本
# ==========================================
if __name__ == '__main__':
    # 假设你的数据集根目录是这个
    base_dir = r'G:\VSCODE-G\PST_Dataset'
    
    # 测试能否正常实例化
    try:
        train_dataset = SARDataset(base_dir=base_dir, domain='DB', mode='train', max_val=255.0)
        
        print(f"✅ 成功加载数据集，共找到 {len(train_dataset)} 个序列切片。")
        
        if len(train_dataset) > 0:
            # 挂载进 DataLoader 测试 Batch
            B = 4
            train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, num_workers=0)
            
            # 取出一个 batch
            inputs, gts = next(iter(train_loader))
            
            print("=" * 40)
            print("📦 批量数据张量维度测试")
            print("=" * 40)
            print(f"👉 Inputs 张量形状: {inputs.shape}  (期望:[B, 2, 16, 512, 512])")
            print(f"👉 GTs   张量形状: {gts.shape}  (期望: [B, 1, 16, 512, 512])")
            print(f"👉 Inputs 值域范围: Min={inputs[:, 0].min():.4f}, Max={inputs[:, 0].max():.4f}")
            print(f"👉 GTs    值域范围: Min={gts.min():.4f}, Max={gts.max():.4f}")
            
    except Exception as e:
        print(f"❌ 测试失败，错误信息: {e}")