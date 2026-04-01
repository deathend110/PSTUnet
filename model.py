import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 模块 1：RAAUNet 空间特征提取基础积木
# ==========================================
class SFEM(nn.Module):
    """
    Shallow Feature Extraction Module (SFEM)
    基于 RAAUNet 的浅层特征提取模块，使用 7x7 -> 5x5 -> 3x3 串联扩大感受野
    """
    def __init__(self, in_channels, out_channels=64):
        super(SFEM, self).__init__()
        
        # 1. 主分支 (Hybrid Convolutional Structure)
        # 严格对应论文公式: Conv(7x7) -> ReLU -> Conv(5x5) -> ReLU -> Conv(3x3) -> ReLU
        # 注意：padding 的设置必须保证特征图宽高不变
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 2. 残差捷径分支 (Residual Connection)
        # 使用 1x1 卷积对齐输入和输出的通道数 (比如把你的 2 通道输入转成 64 通道)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x 的形状:[B*Time, 2, 512, 512] (折叠后的 2D 张量)
        
        # 主分支提取大感受野特征
        main_out = self.main_branch(x)
        
        # 捷径分支保留原始能量基底
        res_out = self.shortcut(x)
        
        # 融合输出 (根据论文图6，相加是在最后一个 ReLU 之后的)
        return self.relu(main_out + res_out)

class CAB(nn.Module):
    """通道注意力"""
    def __init__(self, channels, reduction=16):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid_channels = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class SAB(nn.Module):
    """空间注意力"""
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(out)

class CSAM(nn.Module):
    """串联通道与空间注意力 + 门控过滤"""
    def __init__(self, channels, reduction=16):
        super(CSAM, self).__init__()
        self.cab = CAB(channels, reduction)
        self.sab = SAB(kernel_size=7)
        self.gated_node = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.gated_node(self.sab(self.cab(x)))

class ResBlock(nn.Module):
    """简化版残差块"""
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.convs(x) + x)

class ARCM(nn.Module):
    """注意力残差卷积模块 (替代普通的 DoubleConv)"""
    def __init__(self, in_channels, out_channels, num_blocks):
        super(ARCM, self).__init__()
        self.align_conv = nn.Sequential()
        if in_channels != out_channels:
            self.align_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.ReLU(inplace=True) # 增加非线性映射
            )
        self.csam = CSAM(channels=out_channels, reduction=16)
        self.rg = nn.Sequential(*[ResBlock(out_channels) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.align_conv(x)
        x_att = self.csam(x)
        return self.rg(x_att)

# ==========================================
# 模块 2：RAAUNet 尾端多尺度残差融合
# ==========================================
class RRL(nn.Module):
    """阶段性残差恢复层 (将各层特征压缩为1通道并上采样到原图尺寸)"""
    def __init__(self, in_channels, scale_factor):
        super(RRL, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False)
        if scale_factor > 1:
            self.up = nn.ConvTranspose2d(1, 1, kernel_size=scale_factor, stride=scale_factor, bias=False)
        else:
            self.up = nn.Identity()

    def forward(self, x):
        return self.up(self.conv(x))

class MFRM(nn.Module):
    """多分辨率融合恢复模块"""
    def __init__(self, in_channels=5, out_channels=1):
        super(MFRM, self).__init__()
        self.cab = CAB(in_channels, reduction=max(1, in_channels // 2))
        self.sab = SAB(kernel_size=7)
        self.rhcs_main = nn.Sequential(
            nn.Conv2d(in_channels, 5, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(5, 5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(5, out_channels, kernel_size=7, padding=3)
        )
        self.rhcs_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x_att = self.sab(self.cab(x))
        return self.rhcs_main(x_att) + self.rhcs_shortcut(x_att)

# ==========================================
# 模块 3：原创质量感知时空大动脉 (QAG_PST)
# ==========================================
class QAG_PST_Fusion_Cell(nn.Module):
    """(基于物理切片的) 质量感知残差门控融合单元"""
    def __init__(self, channels, shift_pixels, direction):
        super().__init__()
        self.shift = shift_pixels
        self.direction = direction
        
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels + 1, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_curr, h_prev, conf_curr, conf_prev):
        B, C, H, W = x_curr.shape
        if h_prev is None:
            # 初始时刻无历史记忆，直接拿全零替代
            return self.fusion_conv(torch.cat([x_curr, torch.zeros_like(x_curr)], dim=1))

        q_diff_val = conf_prev - conf_curr 
        
        if self.shift == 0:
            valid_x_curr = x_curr
            valid_h_prev = h_prev
            q_diff_map = q_diff_val.view(B, 1, 1, 1).expand(B, 1, H, W)
        elif self.direction == 'forward':
            valid_x_curr = x_curr[:, :, :, 0 : W - self.shift]
            valid_h_prev = h_prev[:, :, :, self.shift : W]
            q_diff_map = q_diff_val.view(B, 1, 1, 1).expand(B, 1, H, W - self.shift)
        else: # backward
            valid_x_curr = x_curr[:, :, :, self.shift : W]
            valid_h_prev = h_prev[:, :, :, 0 : W - self.shift]
            q_diff_map = q_diff_val.view(B, 1, 1, 1).expand(B, 1, H, W - self.shift)

        # 【神来之笔：恢复基于时序平滑特征的门控逻辑】
        # 虽然 h_prev 和 x_curr 在数学维度上经历了不同的卷积映射，
        # 但 h_prev 是包含了过去多帧信息的“时间平滑态”，极大过滤了 SAR 图像的剧烈散斑噪声 (Speckle)。
        # 拿平滑的 h_prev 去对比充满噪声的 x_curr，能为门控提供远比 x_prev 稳定的时序参考锚点！
        feat_diff = torch.abs(valid_h_prev - valid_x_curr)
        
        # 将物理变化与置信度变化结合，生成软门控
        soft_gate = self.gate_conv(torch.cat([feat_diff, q_diff_map], dim=1)) 
        
        # 用门控去过滤高度抽象的历史记忆
        valid_h_filtered = valid_h_prev * soft_gate

        if self.shift == 0:
            h_aligned = valid_h_filtered
        elif self.direction == 'forward':
            # 填充格式为 (pad_left, pad_right, pad_top, pad_bottom)
            # 填补右侧缺失的像素（新进入视野的区域没有任何记忆，自然填 0）
            h_aligned = F.pad(valid_h_filtered, (0, self.shift, 0, 0)) 
        else:
            # 填补左侧缺失的像素
            h_aligned = F.pad(valid_h_filtered, (self.shift, 0, 0, 0))

        # 将当前特征与过滤后的历史记忆进行终极融合
        return self.fusion_conv(torch.cat([x_curr, h_aligned], dim=1))

class Bi_QAG_PST_Sequence(nn.Module):
    """双向时空渗透网络"""
    def __init__(self, channels, shift_pixels):
        super().__init__()
        self.forward_cell = QAG_PST_Fusion_Cell(channels, shift_pixels, 'forward')
        self.backward_cell = QAG_PST_Fusion_Cell(channels, shift_pixels, 'backward')
        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)
        
        # [Critical Fix] Zero-initialize the fusion layer.
        # This ensures that at the start of training, the Bi_QAG_PST module acts as an identity mapping 
        # (base + 0 = base), meaning it will not perform worse than the baseline model.
        nn.init.constant_(self.fusion.weight, 0)
        if self.fusion.bias is not None:
            nn.init.constant_(self.fusion.bias, 0)

    def forward(self, x_seq, conf_seq):
        B, T, C, H, W = x_seq.shape
        base = x_seq
        
        h_f = None
        out_f = []
        for t in range(T):
            c_curr = conf_seq[:, t] 
            c_prev = conf_seq[:, t-1] if t > 0 else c_curr 
            x_curr = x_seq[:, t]
            
            h_f = self.forward_cell(x_curr, h_f, c_curr, c_prev)
            out_f.append(h_f)
            
        h_b = None
        out_b =[]
        for t in range(T - 1, -1, -1):
            c_curr = conf_seq[:, t]
            c_prev_for_backward = conf_seq[:, t+1] if t < T - 1 else c_curr 
            x_curr = x_seq[:, t]
            
            h_b = self.backward_cell(x_curr, h_b, c_curr, c_prev_for_backward)
            out_b.append(h_b)
        out_b.reverse() # O(N) 且高效，或者使用 out_b = out_b[::-1]
            
        out = []
        for t in range(T):
            combined = torch.cat([out_f[t], out_b[t]], dim=1)
            out.append(self.fusion(combined))
            
        return base + torch.stack(out, dim=1)

# ==========================================
# 终极完全体：PST_UNet 整体架构
# ==========================================
class PST_UNet(nn.Module):
    """
    Physically-aligned Spatio-Temporal U-Net (PST-UNet)
    包含 SFC绿线, RFC黄线, 多尺度MFRM, 以及各层 Skip 的双向时空物理对齐桥接。
    """
    def __init__(self, in_channels=2, out_channels=1, base_dim=64):
        super(PST_UNet, self).__init__()
        # RAAUNet 特性：所有卷积特征层的通道数保持为 base_dim(64)，仅深度改变
        
        # 1. 浅层特征提取 (SFEM)
        self.sfem = SFEM(in_channels=in_channels, out_channels=base_dim)
        
        # 为 SFC 绿线准备的 4 个下采样池化层
        self.sfc_pool1 = nn.MaxPool2d(2)
        self.sfc_pool2 = nn.MaxPool2d(4)
        self.sfc_pool3 = nn.MaxPool2d(8)
        self.sfc_pool4 = nn.MaxPool2d(16)
        
        # 2. 空间编码器 (Encoder)
        # level 0 的输入是原始input+SFC 通道数会变成 64+2
        self.enc0 = ARCM(in_channels=base_dim+in_channels, out_channels=base_dim, num_blocks=1)
        # Level 1~4 的输入是 (上一层池化 + SFC绿线)，通道数会变成 128
        self.enc1 = ARCM(in_channels=base_dim*2, out_channels=base_dim, num_blocks=2)
        self.enc2 = ARCM(in_channels=base_dim*2, out_channels=base_dim, num_blocks=4)
        self.enc3 = ARCM(in_channels=base_dim*2, out_channels=base_dim, num_blocks=8)
        self.enc4 = ARCM(in_channels=base_dim*2, out_channels=base_dim, num_blocks=16) # 瓶颈层
        
        self.pool = nn.MaxPool2d(2) # 编码器层级间的通用下采样

        # 3. 🌟 跨越维度的时空对齐隧道 (Skip Connections)
        # 严格计算每层的物理滑动步长
        self.pst_0 = Bi_QAG_PST_Sequence(channels=base_dim, shift_pixels=128) # 512x512
        self.pst_1 = Bi_QAG_PST_Sequence(channels=base_dim, shift_pixels=64)  # 256x256
        self.pst_2 = Bi_QAG_PST_Sequence(channels=base_dim, shift_pixels=32)  # 128x128
        self.pst_3 = Bi_QAG_PST_Sequence(channels=base_dim, shift_pixels=16)  # 64x64
        self.pst_4 = Bi_QAG_PST_Sequence(channels=base_dim, shift_pixels=8)   # 32x32
        
        # 4. 空间解码器 (Decoder)
        # 上采样层
        self.up4 = nn.ConvTranspose2d(base_dim, base_dim, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(base_dim, base_dim, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_dim, base_dim, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(base_dim, base_dim, kernel_size=2, stride=2)
        
        # Level 4 解码器 (直接接收 pst_4 输出，无需拼接上采样)
        self.dec4 = ARCM(in_channels=base_dim, out_channels=base_dim, num_blocks=16)
        # Level 3~0 解码器 (输入是 本层 PST 跳跃特征 + 上层解码器上采样特征，通道为 128)
        self.dec3 = ARCM(in_channels=base_dim*2, out_channels=base_dim, num_blocks=8)
        self.dec2 = ARCM(in_channels=base_dim*2, out_channels=base_dim, num_blocks=4)
        self.dec1 = ARCM(in_channels=base_dim*2, out_channels=base_dim, num_blocks=2)
        self.dec0 = ARCM(in_channels=base_dim*2, out_channels=base_dim, num_blocks=1)

        # 5. 尾端残差黄线 (RFC) 与 多尺度融合 (MFRM)
        self.rrl_4 = RRL(in_channels=base_dim, scale_factor=16)
        self.rrl_3 = RRL(in_channels=base_dim, scale_factor=8)
        self.rrl_2 = RRL(in_channels=base_dim, scale_factor=4)
        self.rrl_1 = RRL(in_channels=base_dim, scale_factor=2)
        self.rrl_0 = RRL(in_channels=base_dim, scale_factor=1)
        
        self.mfrm = MFRM(in_channels=5, out_channels=out_channels)

    def forward(self, x):
        # x 形状:[B, Channel(2), Time(16), H(512), W(512)]
        B, C, T, H, W = x.shape
        
        # 【物理守恒】：提取原始序列用于残差叠加 (通道 0 是 SAR 图像)
        base_seq = x[:, 0:1, :, :, :] # [B, 1, T, H, W]
        
        # 提取置信度序列 (用于指导各层的 PST-Fusion, 通道 1 是 prompt mask)[B, T]
        conf_seq = x[:, 1, :, 0, 0] 
        
        # 🎓 降维折叠 (Fold): 拍扁成 [B*T, 2, H, W] 以复用 2D 提取器
        x_2d = x.transpose(1, 2).contiguous().view(B * T, C, H, W)
        
        # ==========================================
        # 编码阶段 (含 SFC 绿线逻辑)
        # ==========================================
        # 浅层提取
        sfem_out = self.sfem(x_2d) #[B*T, 64, 512, 512]
        
        # L0 编码
        in_e0 = torch.cat([x_2d, sfem_out], dim=1) #[B*T, 64+C, 512, 512]
        e0 = self.enc0(in_e0) #[B*T, 64, 512, 512]
        
        # L1 编码 (融合下采样与SFC)
        in_e1 = torch.cat([self.pool(e0), self.sfc_pool1(sfem_out)], dim=1) #[B*T, 64*2, 256, 256]
        e1 = self.enc1(in_e1) #[B*T, 64, 256, 256]
        
        # L2 编码
        in_e2 = torch.cat([self.pool(e1), self.sfc_pool2(sfem_out)], dim=1) #[B*T, 64*2, 128, 128]
        e2 = self.enc2(in_e2) #[B*T, 64, 128, 128]
        
        # L3 编码
        in_e3 = torch.cat([self.pool(e2), self.sfc_pool3(sfem_out)], dim=1) #[B*T, 64*2, 64, 64]
        e3 = self.enc3(in_e3) #[B*T, 64, 64, 64]
        
        # L4 编码 (瓶颈层)
        in_e4 = torch.cat([self.pool(e3), self.sfc_pool4(sfem_out)], dim=1) #[B*T, 64*2, 32, 32]
        e4 = self.enc4(in_e4) #[B*T, 64, 32, 32]

        # ==========================================
        # 时空大动脉渗透 (展开为 5D 处理，处理完折叠回 2D)
        # ==========================================
        def apply_pst(feat_2d, pst_module):
            _, c_f, h_f, w_f = feat_2d.shape
            feat_5d = feat_2d.view(B, T, c_f, h_f, w_f)
            fused_5d = pst_module(feat_5d, conf_seq)
            return fused_5d.view(B * T, c_f, h_f, w_f)

        skip0 = apply_pst(e0, self.pst_0)
        skip1 = apply_pst(e1, self.pst_1)
        skip2 = apply_pst(e2, self.pst_2)
        skip3 = apply_pst(e3, self.pst_3)
        skip4 = apply_pst(e4, self.pst_4)

        # ==========================================
        # 解码阶段 (含同帧无损拼接)
        # ==========================================
        d4 = self.dec4(skip4)
        
        in_d3 = torch.cat([skip3, self.up4(d4)], dim=1)
        d3 = self.dec3(in_d3)
        
        in_d2 = torch.cat([skip2, self.up3(d3)], dim=1)
        d2 = self.dec2(in_d2)
        
        in_d1 = torch.cat([skip1, self.up2(d2)], dim=1)
        d1 = self.dec1(in_d1)
        
        in_d0 = torch.cat([skip0, self.up1(d1)], dim=1)
        d0 = self.dec0(in_d0)

        # ==========================================
        # 残差黄线收集 (RFC) 与 多分辨率融合 (MFRM)
        # ==========================================
        res4 = self.rrl_4(d4) # [B*T, 1, 512, 512]
        res3 = self.rrl_3(d3)
        res2 = self.rrl_2(d2)
        res1 = self.rrl_1(d1)
        res0 = self.rrl_0(d0)
        
        multi_res_concat = torch.cat([res0, res1, res2, res3, res4], dim=1) #[B*T, 5, 512, 512]
        
        # 终极残差输出
        residual_2d = self.mfrm(multi_res_concat) # [B*T, 1, 512, 512]

        # ==========================================
        # 全局恢复与输出
        # ==========================================
        residual_seq = residual_2d.view(B, T, 1, H, W).transpose(1, 2) # [B, 1, T, 512, 512]
        restored_seq = base_seq + residual_seq # [B, 1, T, 512, 512]
        
        return restored_seq # [B, 1, T, 512, 512]


# ==========================================
# 维度透视测试
# ==========================================
# ==========================================
# 梯度流与连通性全面测试
# ==========================================
# 1. 在训练开始前，初始化一个 GradScaler
scaler = torch.amp.GradScaler("cuda")
if __name__ == '__main__':
    print("🚀 启动 PST-UNet 梯度反向传播连通性测试...\n")
    
    B, C, T, H, W = 2, 2, 16, 512, 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"👉 模拟输入张量尺寸:[{B}, {C}, {T}, {H}, {W}]")
    print(f"👉 运行设备: {device}\n")

    # 1. 构造 dummy input 和 target
    # 通道0是图像，通道1是掩码
    dummy_input = torch.randn(B, C, T, H, W, device=device)
    dummy_target = torch.randn(B, 1, T, H, W, device=device) # 期望输出是1通道
    
    # 2. 初始化模型
    model = PST_UNet(in_channels=2, out_channels=1, base_dim=64).to(device)

    # 2. 🌟 关键：检测如果有多张卡，就自动开启多卡数据并行
    if torch.cuda.device_count() > 1:
        print(f"🔥 检测到 {torch.cuda.device_count()} 张显卡，启动多卡并行模式！")
        model = nn.DataParallel(model)
    
    # 确保模型处于训练模式（否则Dropout/BatchNorm等行为不同，虽然这里没用到，但保持好习惯）
    model.train()

    # 3. 前向传播
    print("⏳ 正在执行前向传播 (Forward)...")
    with torch.amp.autocast('cuda'): # 使用 AMP 拯救显存
        output = model(dummy_input)
    
    # 4. 计算假定损失 (使用 MSE)
    loss = F.mse_loss(output, dummy_target)
    print(f"✅ 前向传播完成，假定 MSE Loss: {loss.item():.4f}")

    # 5. 反向传播
    print("⏳ 正在执行反向传播 (Backward)...")
    model.zero_grad() # 清空旧梯度
    scaler.scale(loss).backward()
    print("✅ 反向传播完成！\n")

    # 6. 全局梯度检查 (重点核心)
    print("🔍 开始检查各个模块的梯度流...")
    
    has_error = False
    none_grad_params = []
    zero_grad_params =[]
    
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        
        # 检查 1：梯度是否为 None (意味着该参数完全脱离了计算图)
        if param.grad is None:
            none_grad_params.append(name)
            has_error = True
            
        # 检查 2：梯度是否全为 0 (意味着虽然在计算图里，但由于某些截断操作导致梯度没传过来)
        elif torch.sum(torch.abs(param.grad)) == 0:
            zero_grad_params.append(name)
            has_error = True

    # 7. 输出测试报告
    print("=" * 50)
    print("📊 梯度连通性诊断报告")
    print("=" * 50)
    print(f"总检查参数张量数量: {total_params}")
    
    if not has_error:
        print("🎉 测试通过！所有参数都成功接收到了非零梯度！")
        print("说明 PST-UNet 的所有跳跃连接、PST双向模块、注意力机制均完美连通，没有断层。")
    else:
        print("❌ 发现梯度传播异常！")
        if len(none_grad_params) > 0:
            print(f"\n[致命异常] 以下 {len(none_grad_params)} 个参数脱离了计算图 (梯度为 None):")
            for name in none_grad_params:
                print(f"  - {name}")
                
        if len(zero_grad_params) > 0:
            print(f"\n[潜在异常] 以下 {len(zero_grad_params)} 个参数的梯度全为 0 (可能被 ReLU / 绝对值等截断):")
            for name in zero_grad_params:
                print(f"  - {name}")
    print("=" * 50)