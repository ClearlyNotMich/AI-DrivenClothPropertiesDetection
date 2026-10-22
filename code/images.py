"""Split copies of fabric images into tiles without modifying the source data."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import cv2


@dataclass
class TilingResult:
    """Successful source images and tile counts, plus undecodable source files."""

    written: Dict[Path, int] = field(default_factory=dict)
    skipped: List[Path] = field(default_factory=list)


class Data:
    """Preprocess a flat image directory into a separate directory of tiles.

    Defaults retain the original 5000-by-5000 resize and 25-by-25 grid. Paths
    are relative to the caller's working directory unless given absolutely.
    ``processed_data`` is retained for legacy callers; this helper does not
    organize tiles by fabric label or prepare training/validation/test splits.
    """

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(
        self,
        data="data",
        temp_data="data_temp",
        processed_data="data_processed",
        *,
        image_size=5000,
        grid_size=25,
    ):
        for name, value in (("image_size", image_size), ("grid_size", grid_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if image_size % grid_size:
            raise ValueError("image_size must be divisible by grid_size")

        self.data = Path(data).expanduser()
        self.temp_data = Path(temp_data).expanduser()
        self.processed_data = Path(processed_data).expanduser()
        self.image_size = image_size
        self.grid_size = grid_size

    def sq_cut_img(self):
        """Write PNG tiles and return a :class:`TilingResult`.

        Sources are never deleted, moved, or overwritten. Only regular image
        files directly in ``data`` are read; subdirectories and symlinks are
        ignored. Undecodable images are reported in ``result.skipped``.

        Input and output directories must be disjoint. Existing per-image
        output paths and duplicate source stems are rejected before writing.
        Rerunning into an existing output requires choosing a fresh directory.
        Write failures raise an exception; any partial output is retained for
        inspection rather than silently discarded or overwritten.
        """
        source_dir = self.data.resolve()
        output_dir = self.temp_data.resolve()
        if not source_dir.is_dir():
            raise NotADirectoryError("Input directory does not exist: {}".format(source_dir))
        if (
            source_dir == output_dir
            or source_dir in output_dir.parents
            or output_dir in source_dir.parents
        ):
            raise ValueError("Input and output directories must not overlap")
        if self.temp_data.is_symlink():
            raise ValueError("Output directory must not be a symlink")
        if output_dir.exists() and not output_dir.is_dir():
            raise NotADirectoryError("Output path is not a directory: {}".format(output_dir))

        sources = sorted(
            path
            for path in source_dir.iterdir()
            if path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() in self.IMAGE_EXTENSIONS
        )
        if not sources:
            raise ValueError("Input directory contains no supported image files")

        stems = set()
        for source in sources:
            # Case-folding also prevents collisions when output is later moved
            # to a filesystem with case-insensitive names.
            stem = source.stem.casefold()
            if stem in stems:
                raise ValueError("Source images have colliding stems: {}".format(source.stem))
            stems.add(stem)
            destination = output_dir / source.stem
            if destination.exists() or destination.is_symlink():
                raise FileExistsError("Output already exists: {}".format(destination))

        result = TilingResult()
        tile_size = self.image_size // self.grid_size
        for source in sources:
            try:
                image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            except cv2.error:
                image = None
            if image is None:
                result.skipped.append(source)
                continue

            resized = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_CUBIC,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            destination = output_dir / source.stem
            # Exclusive creation guards against an output appearing since the
            # preflight check. No existing directory is reused.
            destination.mkdir()
            for y in range(0, self.image_size, tile_size):
                for x in range(0, self.image_size, tile_size):
                    tile = resized[y : y + tile_size, x : x + tile_size]
                    target = destination / "{}_{}_{}.png".format(source.stem, x, y)
                    if not cv2.imwrite(str(target), tile):
                        raise OSError("Could not write tile: {}".format(target))
            result.written[source] = self.grid_size ** 2

        return result

