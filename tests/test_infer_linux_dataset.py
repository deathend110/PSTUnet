import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


class _FakeH5File:
    def __init__(self, file_map, path, mode="r"):
        self._file_map = file_map
        self._path = os.path.abspath(path)

    def __enter__(self):
        return self._file_map[self._path]

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _load_infer_linux_module(file_map):
    fake_h5py = types.SimpleNamespace(File=lambda path, mode="r": _FakeH5File(file_map, path, mode))
    with mock.patch.dict(sys.modules, {"h5py": fake_h5py}):
        sys.modules.pop("infer_linux", None)
        return importlib.import_module("infer_linux")


class DBInferenceDatasetTests(unittest.TestCase):
    def test_builds_mask_aware_input_from_mode_mask_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            db_dir = base_dir / "DB" / "testdata"
            db_dir.mkdir(parents=True)

            db_path = db_dir / "sample_DB_seq_0001.mat"
            db_path.touch()
            seq_input = np.arange(2 * 3 * 4, dtype=np.float32).reshape(4, 3, 2)
            seq_gt = seq_input + 10.0
            mode_mask = np.array(
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [1.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )

            infer_linux = _load_infer_linux_module(
                {
                    str(db_path.resolve()): {
                        "/seq_input": seq_input,
                        "/seq_GT": seq_gt,
                        "/mode_mask_512_all": mode_mask,
                    }
                }
            )
            dataset = infer_linux.DBInferenceDataset(base_dir=str(base_dir), max_val=255.0)
            inputs, targets, file_name = dataset[0]

            self.assertEqual(file_name, "sample_DB_seq_0001.mat")
            self.assertEqual(tuple(inputs.shape), (2, 2, 4, 3))
            self.assertEqual(tuple(targets.shape), (1, 2, 4, 3))

            expected_seq = seq_input.transpose(2, 0, 1) / 255.0
            np.testing.assert_allclose(inputs[0].numpy(), expected_seq)
            np.testing.assert_allclose(targets[0].numpy(), seq_gt.transpose(2, 0, 1) / 255.0)

            expected_mask = np.broadcast_to(mode_mask[:, np.newaxis, :], (2, 4, 3))
            np.testing.assert_allclose(inputs[1].numpy(), expected_mask)


if __name__ == "__main__":
    unittest.main()
