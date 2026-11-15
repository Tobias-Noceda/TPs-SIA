import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from tp5.parsers.image_folder import fit_image, image_files

import numpy as np
from PIL import Image

def parse_size(text: str) -> tuple[int, int]:
    """`32` → (32, 32); `20x30` → (20, 30) as (height, width)."""
    parts = text.lower().replace("x", "x").split("x")
    try:
        sizes = [int(part) for part in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid size '{text}': use N or HEIGHTxWIDTH.") from None
    if len(sizes) == 1:
        sizes *= 2
    if len(sizes) != 2 or min(sizes) < 1:
        raise argparse.ArgumentTypeError(f"Invalid size '{text}': use N or HEIGHTxWIDTH.")
    return sizes[0], sizes[1]

def groups(source: Path, classes: List[str] | None) -> Dict[str, List[Path]]:
    """
    Images of each class. Classes are the subfolders of `source` that contain images (e.g.
    simpsons_dataset/homer_simpson/); if there are none, every image of `source` is one group (flat
    folder, e.g. one flag per country).
    """
    folders = sorted(path for path in source.iterdir() if path.is_dir() and image_files(path))
    if classes:
        available = {folder.name: folder for folder in folders}
        missing = [name for name in classes if name not in available]
        if missing:
            raise SystemExit(f"Unknown classes: {', '.join(missing)}. Available: {', '.join(available) or '(none: flat folder)'}")
        return {name: image_files(available[name]) for name in classes}
    if folders:
        return {folder.name: image_files(folder) for folder in folders}
    return {"": image_files(source)}

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a small image dataset (subset, resized) for the variational autoencoder from a big image folder.",
    )
    parser.add_argument("-s", "--source", required=True, help="Folder with the original images: one subfolder per class or a flat folder")
    parser.add_argument("-o", "--output", required=True, help="New folder for the dataset (must not exist)")
    parser.add_argument("--classes", nargs="+", help="Subfolders to use (default: all of them)")
    parser.add_argument("--names", nargs="+", help="Only images whose file name (without extension, case-insensitive) is one of these, e.g. argentina brazil")
    parser.add_argument("-n", "--count", type=int, default=None, help="Images picked at random per class (flat folder: in total). Default: all")
    parser.add_argument("--size", type=parse_size, default=(32, 32), help="N or HEIGHTxWIDTH (default 32)")
    parser.add_argument("--fit", choices=["crop", "pad", "stretch"], default="crop", help="Other aspect ratios: crop (default), pad on white or stretch")
    parser.add_argument("--min-side", type=int, default=0, help="Skip images whose smaller side is under this many pixels")
    parser.add_argument("--seed", type=int, default=1, help="Seed of the random pick (default 1)")
    args = parser.parse_args()

    source, output = Path(args.source), Path(args.output)
    if not source.is_dir():
        raise SystemExit(f"Source folder not found: {source}")
    if output.exists():
        raise SystemExit(f"{output} already exists: choose another --output (it is never overwritten).")
    if args.count is not None and args.count < 1:
        raise SystemExit("--count must be positive.")

    height, width = args.size
    random = np.random.default_rng(args.seed)
    names = {name.lower() for name in args.names} if args.names else None

    selected: Dict[str, List[Path]] = {}
    for group, files in groups(source, args.classes).items():
        if names is not None:
            files = [path for path in files if path.stem.lower() in names]
        if args.min_side:
            def big_enough(path: Path) -> bool:
                with Image.open(path) as image:
                    return min(image.size) >= args.min_side
            files = [path for path in files if big_enough(path)]
        if args.count is not None and len(files) > args.count:
            picked = random.choice(len(files), size=args.count, replace=False)
            files = [files[i] for i in sorted(picked)]
        if files:
            selected[group] = files
    if not selected:
        raise SystemExit("No images matched.")

    output.mkdir(parents=True)
    written = 0
    for group, files in selected.items():
        folder = output / group if group else output
        folder.mkdir(parents=True, exist_ok=True)
        for path in files:
            with Image.open(path) as image:
                fit_image(image, height, width, args.fit).save(folder / f"{path.stem}.png")
            written += 1

    # What was done, to rebuild the same dataset
    manifest = {
        "source": str(source),
        "size": [height, width],
        "fit": args.fit,
        "seed": args.seed,
        "count": args.count,
        "classes": args.classes,
        "names": args.names,
        "min_side": args.min_side,
        "images": {group or ".": [path.name for path in files] for group, files in selected.items()},
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    with open(output / "manifest.json", "w") as file:
        json.dump(manifest, file, indent=2)

    print(f"Wrote {written} images ({height}x{width}, {args.fit}) into {output}: "
          + ", ".join(f"{group or '.'} {len(files)}" for group, files in selected.items()))
