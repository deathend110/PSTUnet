import unittest
from pathlib import Path


class TrainAMPSafetySourceTests(unittest.TestCase):
    def test_train_computes_loss_in_fp32_outside_autocast(self):
        """
        这个回归测试约束 AMP 数值安全策略：
        - 模型前向可以继续使用 autocast 提升吞吐
        - 但 SAR 序列的 L1 + TV loss 必须显式切回 FP32 计算
        否则 TV 项在半精度下做平方与求和时，容易在训练中后期出现溢出。
        """
        train_py = Path(__file__).resolve().parents[1] / "train.py"
        source = train_py.read_text(encoding="utf-8")

        self.assertIn("outputs = wrapped_model(inputs)", source)
        self.assertIn("loss = criterion(outputs.float(), targets.float())", source)
        self.assertNotIn("loss = wrapped_model(inputs, targets)", source)

    def test_train_fails_fast_on_non_finite_loss(self):
        """
        这个回归测试约束训练过程中的非有限值保护：
        一旦某个 batch 的 loss 变成 NaN/Inf，必须立刻中止，
        避免继续 step 污染整个模型参数与后续评估结果。
        """
        train_py = Path(__file__).resolve().parents[1] / "train.py"
        source = train_py.read_text(encoding="utf-8")

        self.assertIn("if not torch.isfinite(loss):", source)
        self.assertIn("Non-finite loss detected", source)
        self.assertIn("raise FloatingPointError", source)


if __name__ == "__main__":
    unittest.main()
