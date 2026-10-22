"""Behavioral checks using generated images; no original dataset is required."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "images.py"
SPEC = importlib.util.spec_from_file_location("fabric_images", MODULE_PATH)
IMAGES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMAGES)
Data = IMAGES.Data


class ImageTilingTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "tiles"

    def write_image(self, name="fabric.png"):
        image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        path = self.source / name
        self.assertTrue(cv2.imwrite(str(path), image))
        return path, image

    def processor(self, **kwargs):
        return Data(self.source, self.output, image_size=8, grid_size=2, **kwargs)

    def test_tiles_preserve_pixels_and_original_file(self):
        source, original = self.write_image()
        original_bytes = source.read_bytes()

        result = self.processor().sq_cut_img()

        self.assertEqual(source.read_bytes(), original_bytes)
        self.assertEqual(result.written, {source: 4})
        self.assertEqual(result.skipped, [])
        self.assertEqual(len(list((self.output / "fabric").glob("*.png"))), 4)
        for y in (0, 4):
            for x in (0, 4):
                tile = cv2.imread(str(self.output / "fabric" / "fabric_{}_{}.png".format(x, y)))
                np.testing.assert_array_equal(tile, original[y : y + 4, x : x + 4])

    def test_resize_and_configurable_grid(self):
        source, _ = self.write_image()
        result = Data(self.source, self.output, image_size=12, grid_size=3).sq_cut_img()
        self.assertEqual(result.written[source], 9)
        tiles = list((self.output / "fabric").glob("*.png"))
        self.assertEqual(len(tiles), 9)
        self.assertTrue(all(cv2.imread(str(path)).shape == (4, 4, 3) for path in tiles))

    def test_corrupt_input_is_reported_and_other_images_still_processed(self):
        source, _ = self.write_image()
        corrupt = self.source / "broken.png"
        corrupt.write_bytes(b"not a valid image")
        note = self.source / "notes.txt"
        note.write_text("not an image")

        result = self.processor().sq_cut_img()

        self.assertEqual(result.skipped, [corrupt])
        self.assertEqual(result.written, {source: 4})
        self.assertEqual(corrupt.read_bytes(), b"not a valid image")
        self.assertFalse((self.output / "broken").exists())
        self.assertEqual(note.read_text(), "not an image")

    def test_existing_output_is_untouched_and_no_other_outputs_are_created(self):
        self.write_image("a.png")
        self.write_image("z.png")
        existing = self.output / "z"
        existing.mkdir(parents=True)
        marker = existing / "keep.txt"
        marker.write_text("keep this")

        with self.assertRaises(FileExistsError):
            self.processor().sq_cut_img()

        self.assertEqual(marker.read_text(), "keep this")
        self.assertFalse((self.output / "a").exists())

    def test_duplicate_stems_fail_before_writing(self):
        self.write_image("fabric.png")
        self.write_image("fabric.jpg")
        with self.assertRaises(ValueError):
            self.processor().sq_cut_img()
        self.assertFalse(self.output.exists())

    def test_overlapping_directories_are_rejected(self):
        source, _ = self.write_image()
        original_bytes = source.read_bytes()
        for output in (self.source, self.source / "nested", self.root):
            with self.subTest(output=output), self.assertRaises(ValueError):
                Data(self.source, output, image_size=8, grid_size=2).sq_cut_img()
        self.assertEqual(source.read_bytes(), original_bytes)
        self.assertFalse((self.source / "nested").exists())

    def test_output_symlinks_are_rejected(self):
        self.write_image()
        target = self.root / "other"
        target.mkdir()
        self.output.symlink_to(target, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.processor().sq_cut_img()
        self.assertEqual(list(target.iterdir()), [])

    def test_broken_destination_symlink_is_not_overwritten(self):
        self.write_image()
        self.output.mkdir()
        link = self.output / "fabric"
        link.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            self.processor().sq_cut_img()
        self.assertTrue(link.is_symlink())
        self.assertFalse((self.root / "missing").exists())

    def test_symlinked_images_and_subdirectories_are_not_followed(self):
        source, _ = self.write_image()
        (self.source / "linked.png").symlink_to(source)
        (self.source / "nested").mkdir()
        self.assertTrue(cv2.imwrite(str(self.source / "nested" / "other.png"), np.zeros((8, 8, 3), dtype=np.uint8)))
        result = self.processor().sq_cut_img()
        self.assertEqual(result.written, {source: 4})

    def test_dimensions_must_form_a_positive_even_grid(self):
        for size, grid in ((0, 2), (8, 0), (-8, 2), (8, -2), (7, 2), (8, 9), (8.0, 2), (True, 1), (8, False)):
            with self.subTest(size=size, grid=grid), self.assertRaises(ValueError):
                Data(self.source, self.output, image_size=size, grid_size=grid)
        self.assertFalse(self.output.exists())

    def test_missing_or_empty_input_does_not_create_output(self):
        with self.assertRaises(NotADirectoryError):
            Data(self.root / "missing", self.output).sq_cut_img()
        with self.assertRaises(ValueError):
            self.processor().sq_cut_img()
        self.assertFalse(self.output.exists())

    def test_write_failure_is_observable_and_keeps_original(self):
        source, _ = self.write_image()
        original_bytes = source.read_bytes()
        with patch.object(IMAGES.cv2, "imwrite", return_value=False):
            with self.assertRaises(OSError):
                self.processor().sq_cut_img()
        self.assertEqual(source.read_bytes(), original_bytes)
        self.assertTrue((self.output / "fabric").is_dir())


if __name__ == "__main__":
    unittest.main()
