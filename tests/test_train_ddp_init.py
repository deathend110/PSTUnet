import unittest
from pathlib import Path


class TrainDDPInitSourceTests(unittest.TestCase):
    def test_train_sets_device_before_init_process_group(self):
        """
        这里做源码级回归检查：
        DDP 初始化时必须先绑定当前 rank 对应的 GPU，再初始化 process group。
        对当前 AutoDL vGPU 环境，这个顺序直接影响双卡能否稳定启动。
        """
        train_py = Path(__file__).resolve().parents[1] / "train.py"
        source = train_py.read_text(encoding="utf-8")

        set_device_idx = source.index("torch.cuda.set_device(device_id)")
        init_pg_idx = source.index('dist.init_process_group(backend=args.dist_backend, init_method="env://")')

        self.assertLess(set_device_idx, init_pg_idx)


if __name__ == "__main__":
    unittest.main()
