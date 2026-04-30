import torch
import torch.nn as nn
import torch.nn.functional as F


class SFEM(nn.Module):
    """
    Shallow Feature Extraction Module (SFEM)
    """
    def __init__(self, in_channels, out_channels=64):
        super(SFEM, self).__init__()

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        main_out = self.main_branch(x)
        res_out = self.shortcut(x)
        return self.relu(main_out + res_out)


class CAB(nn.Module):
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
    def __init__(self, channels, reduction=16):
        super(CSAM, self).__init__()
        self.cab = CAB(channels, reduction)
        self.sab = SAB(kernel_size=7)
        self.gated_node = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.gated_node(self.sab(self.cab(x)))


class ResBlock(nn.Module):
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
    def __init__(self, in_channels, out_channels, num_blocks):
        super(ARCM, self).__init__()
        self.align_conv = nn.Sequential()
        if in_channels != out_channels:
            self.align_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.ReLU(inplace=True)
            )
        self.csam = CSAM(channels=out_channels, reduction=16)
        self.rg = nn.Sequential(*[ResBlock(out_channels) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.align_conv(x)
        x_att = self.csam(x)
        return self.rg(x_att)


class RRL(nn.Module):
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


class QAG_PST_Fusion_Cell(nn.Module):
    """
    Mask-aware 质量感知残差门控融合单元
    现在同时利用：
    1. 特征差异 feat_diff
    2. 空间 mask 差异 mask_delta
    3. 帧级 mask 占比差异 conf_delta
    """
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

    def forward(self, x_curr, h_prev, mask_curr, mask_prev, conf_curr, conf_prev):
        B, C, H, W = x_curr.shape

        if h_prev is None:
            return self.fusion_conv(torch.cat([x_curr, torch.zeros_like(x_curr)], dim=1))

        conf_delta_val = torch.abs(conf_prev - conf_curr)  # [B]

        if self.shift == 0:
            valid_x_curr = x_curr
            valid_h_prev = h_prev
            valid_mask_curr = mask_curr
            valid_mask_prev = mask_prev
            conf_delta_map = conf_delta_val.view(B, 1, 1, 1).expand(B, 1, H, W)

        elif self.direction == 'forward':
            valid_x_curr = x_curr[:, :, :, 0: W - self.shift]
            valid_h_prev = h_prev[:, :, :, self.shift: W]

            valid_mask_curr = mask_curr[:, :, :, 0: W - self.shift]
            valid_mask_prev = mask_prev[:, :, :, self.shift: W]

            conf_delta_map = conf_delta_val.view(B, 1, 1, 1).expand(B, 1, H, W - self.shift)

        else:  # backward
            valid_x_curr = x_curr[:, :, :, self.shift: W]
            valid_h_prev = h_prev[:, :, :, 0: W - self.shift]

            valid_mask_curr = mask_curr[:, :, :, self.shift: W]
            valid_mask_prev = mask_prev[:, :, :, 0: W - self.shift]

            conf_delta_map = conf_delta_val.view(B, 1, 1, 1).expand(B, 1, H, W - self.shift)

        feat_diff = torch.abs(valid_h_prev - valid_x_curr)
        mask_delta = torch.abs(valid_mask_prev - valid_mask_curr)

        # 融合空间边界变化 + 帧级全局占比变化
        gate_aux = torch.clamp(mask_delta + 0.5 * conf_delta_map, 0.0, 1.0)

        soft_gate = self.gate_conv(torch.cat([feat_diff, gate_aux], dim=1))
        valid_h_filtered = valid_h_prev * soft_gate

        if self.shift == 0:
            h_aligned = valid_h_filtered
        elif self.direction == 'forward':
            h_aligned = F.pad(valid_h_filtered, (0, self.shift, 0, 0))
        else:
            h_aligned = F.pad(valid_h_filtered, (self.shift, 0, 0, 0))

        return self.fusion_conv(torch.cat([x_curr, h_aligned], dim=1))


class Bi_QAG_PST_Sequence(nn.Module):
    def __init__(self, channels, shift_pixels):
        super().__init__()
        self.forward_cell = QAG_PST_Fusion_Cell(channels, shift_pixels, 'forward')
        self.backward_cell = QAG_PST_Fusion_Cell(channels, shift_pixels, 'backward')
        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)

        nn.init.constant_(self.fusion.weight, 0)
        if self.fusion.bias is not None:
            nn.init.constant_(self.fusion.bias, 0)

    def forward(self, x_seq, mask_seq, conf_seq):
        """
        x_seq   : [B, T, C, H, W]
        mask_seq: [B, T, 1, H, W]
        conf_seq: [B, T]
        """
        B, T, C, H, W = x_seq.shape
        base = x_seq

        h_f = None
        out_f = []
        for t in range(T):
            x_curr = x_seq[:, t]
            mask_curr = mask_seq[:, t]
            conf_curr = conf_seq[:, t]

            if t > 0:
                mask_prev = mask_seq[:, t - 1]
                conf_prev = conf_seq[:, t - 1]
            else:
                mask_prev = mask_curr
                conf_prev = conf_curr

            h_f = self.forward_cell(x_curr, h_f, mask_curr, mask_prev, conf_curr, conf_prev)
            out_f.append(h_f)

        h_b = None
        out_b = []
        for t in range(T - 1, -1, -1):
            x_curr = x_seq[:, t]
            mask_curr = mask_seq[:, t]
            conf_curr = conf_seq[:, t]

            if t < T - 1:
                mask_prev_for_backward = mask_seq[:, t + 1]
                conf_prev_for_backward = conf_seq[:, t + 1]
            else:
                mask_prev_for_backward = mask_curr
                conf_prev_for_backward = conf_curr

            h_b = self.backward_cell(
                x_curr, h_b,
                mask_curr, mask_prev_for_backward,
                conf_curr, conf_prev_for_backward
            )
            out_b.append(h_b)

        out_b.reverse()

        out = []
        for t in range(T):
            combined = torch.cat([out_f[t], out_b[t]], dim=1)
            out.append(self.fusion(combined))

        return base + torch.stack(out, dim=1)


class PST_UNet(nn.Module):
    """
    Mask-aware PST-UNet
    输入:
        通道0 -> degraded image
        通道1 -> spatial mode mask
    """
    def __init__(self, in_channels=2, out_channels=1, base_dim=64):
        super(PST_UNet, self).__init__()

        self.sfem = SFEM(in_channels=in_channels, out_channels=base_dim)

        self.sfc_pool1 = nn.MaxPool2d(2)
        self.sfc_pool2 = nn.MaxPool2d(4)
        self.sfc_pool3 = nn.MaxPool2d(8)
        self.sfc_pool4 = nn.MaxPool2d(16)

        self.enc0 = ARCM(in_channels=base_dim + in_channels, out_channels=base_dim, num_blocks=1)
        self.enc1 = ARCM(in_channels=base_dim * 2, out_channels=base_dim, num_blocks=2)
        self.enc2 = ARCM(in_channels=base_dim * 2, out_channels=base_dim, num_blocks=4)
        self.enc3 = ARCM(in_channels=base_dim * 2, out_channels=base_dim, num_blocks=8)
        self.enc4 = ARCM(in_channels=base_dim * 2, out_channels=base_dim, num_blocks=16)

        self.pool = nn.MaxPool2d(2)

        self.pst_0 = Bi_QAG_PST_Sequence(channels=base_dim, shift_pixels=128)
        self.pst_1 = Bi_QAG_PST_Sequence(channels=base_dim, shift_pixels=64)
        self.pst_2 = Bi_QAG_PST_Sequence(channels=base_dim, shift_pixels=32)
        self.pst_3 = Bi_QAG_PST_Sequence(channels=base_dim, shift_pixels=16)
        self.pst_4 = Bi_QAG_PST_Sequence(channels=base_dim, shift_pixels=8)

        self.up4 = nn.ConvTranspose2d(base_dim, base_dim, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(base_dim, base_dim, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_dim, base_dim, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(base_dim, base_dim, kernel_size=2, stride=2)

        self.dec4 = ARCM(in_channels=base_dim, out_channels=base_dim, num_blocks=16)
        self.dec3 = ARCM(in_channels=base_dim * 2, out_channels=base_dim, num_blocks=8)
        self.dec2 = ARCM(in_channels=base_dim * 2, out_channels=base_dim, num_blocks=4)
        self.dec1 = ARCM(in_channels=base_dim * 2, out_channels=base_dim, num_blocks=2)
        self.dec0 = ARCM(in_channels=base_dim * 2, out_channels=base_dim, num_blocks=1)

        self.rrl_4 = RRL(in_channels=base_dim, scale_factor=16)
        self.rrl_3 = RRL(in_channels=base_dim, scale_factor=8)
        self.rrl_2 = RRL(in_channels=base_dim, scale_factor=4)
        self.rrl_1 = RRL(in_channels=base_dim, scale_factor=2)
        self.rrl_0 = RRL(in_channels=base_dim, scale_factor=1)

        self.mfrm = MFRM(in_channels=5, out_channels=out_channels)

    def forward(self, x):
        # x: [B, 2, T, H, W]
        B, C, T, H, W = x.shape

        base_seq = x[:, 0:1, :, :, :]  # [B,1,T,H,W]

        # 第二通道是空间 mask
        mask_seq_5d = x[:, 1:2, :, :, :].transpose(1, 2).contiguous()  # [B,T,1,H,W]

        # 帧级全局置信度：每帧高采样占比
        conf_seq = mask_seq_5d.mean(dim=(-1, -2)).squeeze(-1)  # [B,T]

        # [B*T, 2, H, W]
        x_2d = x.transpose(1, 2).contiguous().view(B * T, C, H, W)

        # [B*T, 1, H, W]
        mask_2d = mask_seq_5d.contiguous().view(B * T, 1, H, W)

        sfem_out = self.sfem(x_2d)

        in_e0 = torch.cat([x_2d, sfem_out], dim=1)
        e0 = self.enc0(in_e0)

        in_e1 = torch.cat([self.pool(e0), self.sfc_pool1(sfem_out)], dim=1)
        e1 = self.enc1(in_e1)

        in_e2 = torch.cat([self.pool(e1), self.sfc_pool2(sfem_out)], dim=1)
        e2 = self.enc2(in_e2)

        in_e3 = torch.cat([self.pool(e2), self.sfc_pool3(sfem_out)], dim=1)
        e3 = self.enc3(in_e3)

        in_e4 = torch.cat([self.pool(e3), self.sfc_pool4(sfem_out)], dim=1)
        e4 = self.enc4(in_e4)

        def apply_pst(feat_2d, pst_module):
            _, c_f, h_f, w_f = feat_2d.shape

            feat_5d = feat_2d.view(B, T, c_f, h_f, w_f)

            mask_resized_2d = F.interpolate(mask_2d, size=(h_f, w_f), mode='nearest')
            mask_resized_5d = mask_resized_2d.view(B, T, 1, h_f, w_f)

            fused_5d = pst_module(feat_5d, mask_resized_5d, conf_seq)
            return fused_5d.view(B * T, c_f, h_f, w_f)

        skip0 = apply_pst(e0, self.pst_0)
        skip1 = apply_pst(e1, self.pst_1)
        skip2 = apply_pst(e2, self.pst_2)
        skip3 = apply_pst(e3, self.pst_3)
        skip4 = apply_pst(e4, self.pst_4)

        d4 = self.dec4(skip4)

        in_d3 = torch.cat([skip3, self.up4(d4)], dim=1)
        d3 = self.dec3(in_d3)

        in_d2 = torch.cat([skip2, self.up3(d3)], dim=1)
        d2 = self.dec2(in_d2)

        in_d1 = torch.cat([skip1, self.up2(d2)], dim=1)
        d1 = self.dec1(in_d1)

        in_d0 = torch.cat([skip0, self.up1(d1)], dim=1)
        d0 = self.dec0(in_d0)

        res4 = self.rrl_4(d4)
        res3 = self.rrl_3(d3)
        res2 = self.rrl_2(d2)
        res1 = self.rrl_1(d1)
        res0 = self.rrl_0(d0)

        multi_res_concat = torch.cat([res0, res1, res2, res3, res4], dim=1)
        residual_2d = self.mfrm(multi_res_concat)

        residual_seq = residual_2d.view(B, T, 1, H, W).transpose(1, 2)
        restored_seq = base_seq + residual_seq

        return restored_seq


if __name__ == '__main__':
    print("🚀 启动 Mask-aware PST-UNet 梯度反向传播连通性测试...\n")

    B, C, T, H, W = 1, 2, 4, 512, 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"👉 模拟输入张量尺寸:[{B}, {C}, {T}, {H}, {W}]")
    print(f"👉 运行设备: {device}\n")

    dummy_input = torch.randn(B, C, T, H, W, device=device)
    dummy_target = torch.randn(B, 1, T, H, W, device=device)

    # 构造二值空间 mask，更贴近真实输入
    dummy_input[:, 1] = (dummy_input[:, 1] > 0).float()

    model = PST_UNet(in_channels=2, out_channels=1, base_dim=64).to(device)

    if torch.cuda.device_count() > 1:
        print(f"🔥 检测到 {torch.cuda.device_count()} 张显卡，启动多卡并行模式！")
        model = nn.DataParallel(model)

    model.train()

    # 仅在本地梯度连通性调试分支中创建 AMP scaler。
    # 正常训练时 import model.py 不应提前触发 CUDA/AMP 初始化，
    # 否则在 DDP 的 LOCAL_RANK 绑定设备前就可能引入底层状态副作用。
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    print("⏳ 正在执行前向传播 (Forward)...")
    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        output = model(dummy_input)

    loss = F.mse_loss(output, dummy_target)
    print(f"✅ 前向传播完成，假定 MSE Loss: {loss.item():.4f}")

    print("⏳ 正在执行反向传播 (Backward)...")
    model.zero_grad()
    scaler.scale(loss).backward()
    print("✅ 反向传播完成！\n")

    print("🔍 开始检查各个模块的梯度流...")

    has_error = False
    none_grad_params = []
    zero_grad_params = []
    total_params = 0

    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is None:
            none_grad_params.append(name)
            has_error = True
        elif torch.sum(torch.abs(param.grad)) == 0:
            zero_grad_params.append(name)
            has_error = True

    print("=" * 50)
    print("📊 梯度连通性诊断报告")
    print("=" * 50)
    print(f"总检查参数张量数量: {total_params}")

    if not has_error:
        print("🎉 测试通过！所有参数都成功接收到了非零梯度！")
    else:
        print("❌ 发现梯度传播异常！")
        if len(none_grad_params) > 0:
            print(f"\n[致命异常] 以下 {len(none_grad_params)} 个参数脱离了计算图 (梯度为 None):")
            for name in none_grad_params:
                print(f"  - {name}")

        if len(zero_grad_params) > 0:
            print(f"\n[潜在异常] 以下 {len(zero_grad_params)} 个参数的梯度全为 0:")
            for name in zero_grad_params:
                print(f"  - {name}")

    print("=" * 50)
