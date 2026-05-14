import unittest
from pathlib import Path


class RunAutoInferShellScriptTests(unittest.TestCase):
    def test_shell_script_exists_and_calls_auto_inference(self):
        script_path = Path("run_autodl_infer_all.sh")
        self.assertTrue(script_path.is_file(), "run_autodl_infer_all.sh should exist")

        script_text = script_path.read_text(encoding="utf-8")
        self.assertIn("python infer_linux.py", script_text)
        self.assertIn("--auto-run-all", script_text)
        self.assertIn("--checkpoints-dir", script_text)
        self.assertIn("--dataset-root", script_text)
        self.assertIn("--inference-root", script_text)


if __name__ == "__main__":
    unittest.main()
