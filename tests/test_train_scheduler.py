import unittest
from pathlib import Path


class TrainSchedulerSourceTests(unittest.TestCase):
    def test_train_uses_cosine_annealing_scheduler(self):
        """
        由于当前测试环境不一定安装 torch，这里先做源码级回归校验：
        训练入口必须显式使用余弦退火调度器，而不能继续依赖旧的分段降学习率逻辑。
        """
        train_py = Path(__file__).resolve().parents[1] / "train.py"
        source = train_py.read_text(encoding="utf-8")

        self.assertIn("CosineAnnealingLR", source)
        self.assertIn("scheduler.step()", source)
        self.assertIn("eta_min=1e-6", source)
        self.assertNotIn('schedule = get_epoch_schedule(epoch, base_lr)', source)


if __name__ == "__main__":
    unittest.main()
