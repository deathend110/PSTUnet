import torch
import torch.nn as nn
from model import PST_UNet

def check_grad(name, tensor):
    """梯度探针辅助函数"""
    if tensor is None:
        print(f"❌ [断裂] {name}: 张量为空 (None)！")
        return False
    elif tensor.grad is None:
        print(f"❌ [断裂] {name}: 梯度为 None！该模块未参与反向传播，被孤立了！")
        return False
    elif torch.sum(torch.abs(tensor.grad)) == 0:
        print(f"⚠️ [警报] {name}: 梯度全为 0！遭遇极其严重的梯度消失或硬截断！")
        return False
    else:
        # 计算平均梯度绝对值，反映该模块的学习活跃度
        mean_grad = torch.mean(torch.abs(tensor.grad)).item()
        print(f"✅ [通畅] {name} | 平均梯度活跃度: {mean_grad:.6f}")
        return True

def main():
    print("="*60)
    print("🚀 启动 DST-UNet 全网梯度连通性 X光扫描")
    print("="*60)

    # 1. 初始化模型与伪造数据
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = PST_UNet(in_channels=2, out_channels=1).to(device)
    
    # 模拟 Batch=1, Channel=2, Time=16, H=512, W=512
    # 必须设置 requires_grad=True 才能追踪到最源头的输入
    dummy_input = torch.randn(1, 2, 16, 512, 512, device=device, requires_grad=True)
    dummy_target = torch.randn(1, 1, 16, 512, 512, device=device)

    print("🧠 正在执行前向传播 (Forward Pass)...")
    # 为了看到最纯粹的数学梯度，我们这里不开启 AMP 混合精度
    # output = model(dummy_input)
    
    with torch.amp.autocast('cuda'): # 使用 AMP 拯救显存
        output = model(dummy_input)

    print("🧨 正在执行反向传播 (Backward Pass & BPTT)...")
    # 使用一个简单的 L1 Loss 触发梯度回传
    criterion = nn.L1Loss()
    loss = criterion(output, dummy_target)
    loss.backward()

    print("\n" + "="*60)
    print("📡 核心模块梯度连通性诊断报告")
    print("="*60)

    # ==========================================
    # 1. 检查输入端 (最源头，看长序列梯度是否消失)
    # ==========================================
    print("【第一关：输入端源头】")
    check_grad("Input Tensor (输入图像与掩码)", dummy_input)
    check_grad("SFEM.main_branch (浅层多尺度特征提取)", model.sfem.main_branch[0].weight)

    # ==========================================
    # 2. 检查你魔改的 ARCM (注意力与残差)
    # ==========================================
    print("\n【第二关：ARCM 空间雕刻器】")
    # 检查 Level 0 的 Encoder (这里有你神来之笔的 Dense Connection 拼接)
    check_grad("Enc0_ARCM.csam.cab (通道注意力 MLP)", model.enc0.csam.cab.mlp[0].weight)
    check_grad("Enc0_ARCM.csam.sab (空间注意力大核)", model.enc0.csam.sab.conv.weight)
    check_grad("Enc0_ARCM.rg (残差重装部队)", model.enc0.rg[0].convs[0].weight)

    # ==========================================
    # 3. 检查最复杂的时空大动脉 (PST-Sequence)
    # ==========================================
    print("\n【第三关：PST-Fusion 时空大动脉】")
    # 我们抽查最深处的 pst_4 (32x32 瓶颈层，偏移 8 像素) 和中间的 pst_2 (128x128，偏移 32 像素)
    check_grad("PST4.Forward_Cell.SoftGate (正向残差软门控)", model.pst_4.forward_cell.gate_conv[0].weight)
    check_grad("PST4.Backward_Cell.SoftGate (反向残差软门控)", model.pst_4.backward_cell.gate_conv[0].weight)
    check_grad("PST4.Bidirectional_Fusion (双向会师降维 1x1)", model.pst_4.bidirectional_fusion.weight)
    
    check_grad("PST2.Forward_Cell.SoftGate (中层正向残差软门控)", model.pst_2.forward_cell.gate_conv[0].weight)

    # ==========================================
    # 4. 检查多尺度残差尾部 (MFRM)
    # ==========================================
    print("\n【第四关：RRL 与 MFRM 尾端融合】")
    check_grad("RRL_4 (最深层特征 16 倍上采样残差提取)", model.rrl_4.conv.weight)
    check_grad("RRL_0 (最浅层特征原尺寸残差提取)", model.rrl_0.conv.weight)
    check_grad("MFRM.rhcs_main (5大残差图多尺度终极融合)", model.mfrm.rhcs_main[0].weight)

    print("\n" + "="*60)
    print("🎉 诊断完成！如果全是绿色的 ✅，说明你的网络是一个完美的物理闭环！")

if __name__ == '__main__':
    main()