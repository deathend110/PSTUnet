import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SARDataset(Dataset):
    def __init__(self, base_dir, domain='DB', mode='train', max_val=255.0):
        """
        SAR 序列数据集加载器（适配新的 AzimuthMix 序列数据集）

        参数：
            base_dir : 数据集根目录，例如:
                       /root/autodl-tmp/Sequence_Dataset_AzimuthMix_q3_rt_only
            domain   : 'DB' 或 'Linear'
            mode     : 'train' 或 'test'
            max_val  : 反量化除数。uint8 图像填 255.0；若以后改成 uint16 则填 65535.0
        """
        self.domain = domain
        self.max_val = max_val

        folder_mode = 'traindata' if mode == 'train' else 'testdata'
        self.data_dir = os.path.join(base_dir, domain, folder_mode)

        self.file_paths = sorted(glob.glob(os.path.join(self.data_dir, '*.mat')))
        if len(self.file_paths) == 0:
            print(f"⚠️ 警告: 在 {self.data_dir} 下没有找到任何 .mat 文件，请检查路径。")

        if self.domain == 'DB':
            self.key_in = '/seq_input'
            self.key_gt = '/seq_GT'
        elif self.domain == 'Linear':
            self.key_in = '/seq_input_L'
            self.key_gt = '/seq_GT_L'
        else:
            raise ValueError("domain 参数必须是 'DB' 或 'Linear'")

        self.key_frame_mode_id = '/frame_mode_id'
        self.key_mode_mask_512_all = '/mode_mask_512_all'
        self.key_sigma_seq = '/sigma_seq'
        self.key_A_rt = '/A_rt'

    def __len__(self):
        return len(self.file_paths)

    @staticmethod
    def _to_thw(arr: np.ndarray) -> np.ndarray:
        """
        尽量稳妥地把输入转成 [T, H, W]
        当前你的数据是 512x512，H/W 相同，因此旧版 transpose(0,2,1) 不会出错。
        这里做一个更稳妥的兼容写法。
        """
        if arr.ndim != 3:
            raise ValueError(f"期望 3D 数组，实际 shape={arr.shape}")

        # 常见情况1: [H, W, T]
        if arr.shape[0] > 64 and arr.shape[1] > 64 and arr.shape[2] <= 64:
            return arr.transpose(2, 0, 1)

        # 常见情况2: [T, H, W] 或 [T, W, H]
        if arr.shape[0] <= 64 and arr.shape[1] > 64 and arr.shape[2] > 64:
            # 对你当前 512x512 的数据来说 transpose 与否都不影响 H/W
            return arr

        raise ValueError(f"无法识别序列维度顺序，shape={arr.shape}")

    @staticmethod
    def _frame_mode_id_to_scalar(frame_mode_id: np.ndarray) -> np.ndarray:
        """
        frame_mode_id 编码:
            low   -> 1
            high  -> 2
            mixed -> 3
            gt    -> 4

        这里给一个兼容的帧级标量表征，可用于回退或调试：
            low   -> 0.0
            mixed -> 0.5
            high  -> 1.0
            gt    -> 1.0
        """
        frame_mode_id = frame_mode_id.astype(np.int64).flatten()
        scalar = np.zeros_like(frame_mode_id, dtype=np.float32)
        scalar[frame_mode_id == 1] = 0.0
        scalar[frame_mode_id == 2] = 1.0
        scalar[frame_mode_id == 3] = 0.5
        scalar[frame_mode_id == 4] = 1.0
        return scalar

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        with h5py.File(file_path, 'r') as f:
            seq_in = np.array(f[self.key_in]).astype(np.float32)
            seq_gt = np.array(f[self.key_gt]).astype(np.float32)

            if self.key_frame_mode_id in f:
                frame_mode_id = np.array(f[self.key_frame_mode_id]).astype(np.uint8).reshape(-1)
            else:
                frame_mode_id = None

            if self.key_mode_mask_512_all in f:
                mode_mask_512_all = np.array(f[self.key_mode_mask_512_all]).astype(np.float32)
            else:
                mode_mask_512_all = None

            # 先读出来，当前版本不直接返回，但保留兼容
            if self.key_sigma_seq in f:
                sigma_seq = np.array(f[self.key_sigma_seq]).astype(np.float32).reshape(-1)
            else:
                sigma_seq = None

            if self.key_A_rt in f:
                A_rt = np.array(f[self.key_A_rt]).astype(np.float32).reshape(-1)
            else:
                A_rt = None

        seq_in = self._to_thw(seq_in)
        seq_gt = self._to_thw(seq_gt)

        seq_in = seq_in / self.max_val
        seq_gt = seq_gt / self.max_val

        seq_in = np.clip(seq_in, 0.0, 1.0)
        seq_gt = np.clip(seq_gt, 0.0, 1.0)

        T, H, W = seq_in.shape

        # ----------------------------
        # 构建第二通道：真实空间 mask
        # mode_mask_512_all 存的是 [T, W] 或 [W, T]
        # 这里扩展成 [T, H, W]
        # ----------------------------
        if mode_mask_512_all is not None:
            if mode_mask_512_all.ndim != 2:
                raise ValueError(
                    f"{file_path} 中 {self.key_mode_mask_512_all} 维度异常: shape={mode_mask_512_all.shape}"
                )

            if mode_mask_512_all.shape[0] == T and mode_mask_512_all.shape[1] == W:
                mode_mask_tw = mode_mask_512_all
            elif mode_mask_512_all.shape[1] == T and mode_mask_512_all.shape[0] == W:
                mode_mask_tw = mode_mask_512_all.T
            else:
                raise ValueError(
                    f"{file_path} 中 {self.key_mode_mask_512_all} 与图像尺寸不匹配: "
                    f"mask shape={mode_mask_512_all.shape}, seq shape={seq_in.shape}"
                )

            mask_seq = np.broadcast_to(
                mode_mask_tw[:, np.newaxis, :], (T, H, W)
            ).astype(np.float32)

        else:
            # 回退方案：如果没有 mode_mask_512_all，就退化成每帧常值 mask
            if frame_mode_id is None:
                raise KeyError(f"{file_path} 中既没有 {self.key_mode_mask_512_all}，也没有 {self.key_frame_mode_id}")

            frame_scalar = self._frame_mode_id_to_scalar(frame_mode_id)
            if len(frame_scalar) != T:
                raise ValueError(
                    f"{file_path} 中 frame_mode_id 长度({len(frame_scalar)}) 与序列帧数 T({T}) 不一致"
                )
            mask_seq = np.broadcast_to(
                frame_scalar[:, np.newaxis, np.newaxis], (T, H, W)
            ).astype(np.float32)

        input_tensor = np.stack([seq_in, mask_seq], axis=0).astype(np.float32)
        gt_tensor = seq_gt[np.newaxis, :, :, :].astype(np.float32)

        return torch.from_numpy(input_tensor), torch.from_numpy(gt_tensor)


if __name__ == '__main__':
    base_dir = r'G:\VSCODE-G\Sequence_Dataset_AzimuthMix_q3_rt_only'

    try:
        train_dataset = SARDataset(base_dir=base_dir, domain='DB', mode='train', max_val=255.0)
        print(f"✅ 成功加载数据集，共找到 {len(train_dataset)} 个序列样本。")

        if len(train_dataset) > 0:
            loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
            inputs, gts = next(iter(loader))

            print("=" * 50)
            print("📦 批量数据张量维度测试")
            print("=" * 50)
            print(f"👉 Inputs 张量形状: {inputs.shape}  (期望: [B, 2, T, 512, 512])")
            print(f"👉 GTs   张量形状: {gts.shape}    (期望: [B, 1, T, 512, 512])")
            print(f"👉 Inputs[img] 值域: Min={inputs[:, 0].min():.4f}, Max={inputs[:, 0].max():.4f}")
            print(f"👉 Inputs[mask] 值域: Min={inputs[:, 1].min():.4f}, Max={inputs[:, 1].max():.4f}")
            print(f"👉 GTs 值域: Min={gts.min():.4f}, Max={gts.max():.4f}")

    except Exception as e:
        print(f"❌ 测试失败，错误信息: {e}")