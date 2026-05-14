import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


def _build_fake_modules():
    fake_torch = types.ModuleType("torch")
    fake_torch.device = lambda text: text
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    fake_utils_data = types.ModuleType("torch.utils.data")

    class _Dataset:
        pass

    fake_utils_data.Dataset = _Dataset
    fake_utils_data.DataLoader = object

    fake_model = types.ModuleType("model")
    fake_model.PST_UNet = object

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.SARDataset = object

    fake_utils = types.ModuleType("utils")
    fake_utils.SSIMLoss = object
    fake_utils.calc_psnr = object

    fake_numpy = types.ModuleType("numpy")

    fake_scipy = types.ModuleType("scipy")
    fake_scipy_io = types.ModuleType("scipy.io")
    fake_scipy.io = fake_scipy_io

    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = object

    return {
        "torch": fake_torch,
        "torch.utils.data": fake_utils_data,
        "model": fake_model,
        "datasets": fake_datasets,
        "utils": fake_utils,
        "numpy": fake_numpy,
        "scipy": fake_scipy,
        "scipy.io": fake_scipy_io,
        "tqdm": fake_tqdm,
    }


class InferLinuxAutoRunTests(unittest.TestCase):
    def _load_module(self):
        fake_modules = _build_fake_modules()
        with mock.patch.dict(sys.modules, fake_modules):
            sys.modules.pop("infer_linux", None)
            return importlib.import_module("infer_linux")

    def test_build_auto_inference_jobs_from_output_checkpoints(self):
        infer_linux = self._load_module()

        checkpoint_paths = [
            Path("./output/Model(PST_UNet)-Dataset(AzimuthMix_q3)-Loss(x)-domainLinear/best.pth"),
            Path("./output/Model(PST_UNet)-Dataset(RangeMix_q7)-Loss(x)-domainLinear/best.pth"),
        ]

        with mock.patch.object(infer_linux.Path, "glob", return_value=checkpoint_paths):
            jobs = infer_linux.build_auto_inference_jobs(
                checkpoints_dir="./output",
                dataset_root="/root/autodl-tmp",
                inference_root="./inference_output",
            )

        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]["dataset_name"], "AzimuthMix_q3")
        self.assertEqual(jobs[0]["base_dir"], "/root/autodl-tmp/Sequence_Dataset_AzimuthMix_q3_rt_only")
        self.assertEqual(jobs[0]["output_dir"], "inference_output/linear_test_AzimuthMix_q3")
        self.assertEqual(jobs[1]["dataset_name"], "RangeMix_q7")
        self.assertEqual(jobs[1]["base_dir"], "/root/autodl-tmp/Sequence_Dataset_RangeMix_q7_rt_only")
        self.assertEqual(jobs[1]["output_dir"], "inference_output/linear_test_RangeMix_q7")

    def test_extract_dataset_name_from_checkpoint_requires_dataset_marker(self):
        infer_linux = self._load_module()

        with self.assertRaisesRegex(ValueError, "Dataset\\("):
            infer_linux.extract_dataset_name_from_checkpoint(
                "./output/Model(PST_UNet)-Loss(x)-domainLinear/best.pth"
            )


if __name__ == "__main__":
    unittest.main()
