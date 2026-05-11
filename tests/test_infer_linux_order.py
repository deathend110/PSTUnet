import importlib
import sys
import types
import unittest
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


class LinearInferenceOrderTests(unittest.TestCase):
    def _load_module(self):
        fake_modules = _build_fake_modules()
        with mock.patch.dict(sys.modules, fake_modules):
            sys.modules.pop("infer_linux", None)
            return importlib.import_module("infer_linux")

    def test_linear_inference_dataset_returns_stable_sample_index(self):
        infer_linux = self._load_module()

        fake_dataset = mock.Mock()
        fake_dataset.data_dir = "/tmp/Linear/testdata"
        fake_dataset.file_paths = [
            "/tmp/Linear/testdata/airport_L_seq_0003.mat",
            "/tmp/Linear/testdata/city_L_seq_0001.mat",
            "/tmp/Linear/testdata/harbor_L_seq_0002.mat",
        ]
        fake_dataset.__getitem__ = mock.Mock(
            side_effect=[
                ("input0", "target0"),
                ("input1", "target1"),
                ("input2", "target2"),
            ]
        )

        with mock.patch.object(infer_linux, "SARDataset", return_value=fake_dataset):
            dataset = infer_linux.LinearInferenceDataset(base_dir="/tmp")

        sample0 = dataset[0]
        sample1 = dataset[1]
        sample2 = dataset[2]

        self.assertEqual(sample0, ("input0", "target0", "airport_L_seq_0003.mat", 0))
        self.assertEqual(sample1, ("input1", "target1", "city_L_seq_0001.mat", 1))
        self.assertEqual(sample2, ("input2", "target2", "harbor_L_seq_0002.mat", 2))


if __name__ == "__main__":
    unittest.main()
