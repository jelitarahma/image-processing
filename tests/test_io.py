"""Unit tests for image loading and saving utilities."""

from pathlib import Path
import numpy as np
import pytest

from illumination_lab.io import load_image, save_image


def test_load_nonexistent_file(tmp_path: Path) -> None:
    non_existent = tmp_path / "does_not_exist.png"
    with pytest.raises(FileNotFoundError):
        load_image(non_existent)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    dummy_img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    out_file = tmp_path / "test_out.png"

    saved_path = save_image(out_file, dummy_img)
    assert saved_path.is_file()

    loaded = load_image(saved_path)
    assert loaded.shape == dummy_img.shape
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, dummy_img)


def test_save_empty_image(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_image(tmp_path / "empty.png", np.array([]))
