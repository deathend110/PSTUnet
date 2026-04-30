import unittest
from pathlib import Path


class ModelImportSafetyTests(unittest.TestCase):
    def test_model_module_does_not_create_grad_scaler_at_import_time(self):
        """
        DDP 场景下，模型模块在 import 阶段不应主动创建 CUDA/AMP 对象，
        否则可能在 LOCAL_RANK 绑定设备前触发底层初始化。
        这里用源码级约束，防止再次引入顶层 GradScaler 副作用。
        """
        model_py = Path(__file__).resolve().parents[1] / "model.py"
        source = model_py.read_text(encoding="utf-8")

        self.assertNotIn('scaler = torch.amp.GradScaler("cuda")', source)


if __name__ == "__main__":
    unittest.main()
