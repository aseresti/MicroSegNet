#!/usr/bin/env python3
"""Turn the Zenodo micro-ultrasound tree into nnU-Net raw dataset 810.

The Zenodo record is train/ and test/, each with micro_ultrasound_scans/ and
expert_annotations/. Names do not match (microUS_train_01 vs
expert_annotation_train_01), and the test folder also holds other annotators.
nnU-Net needs one stem per case, channel suffix _0000 on the image only, and
integer labels 0 and 1.

This script pairs on the case id (the last two underscore pieces: train_01),
copies the image bytes unchanged, and rewrites the expert mask as 0/1 with the
image geometry. Non-expert, clinician, and student folders are ignored.
The 20 test patients go to imagesTs. Their expert masks go beside the dataset,
where training does not read them.

Default is a dry run. Pass --write to create files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


DATASET_NAME = "Dataset810_MicroUSProstate"


def case_id_from_name(filename: str) -> str:
    name = str(filename).replace("\\", "/").split("/")[-1]
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(f"cannot read a case id from {filename}")
    return "_".join(parts[-2:])


def list_volumes(folder: Path) -> list[Path]:
    files = [
        p
        for p in folder.iterdir()
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
    ]
    return sorted(files, key=lambda p: p.name)


def pair_split(scans: Path, experts: Path) -> list[tuple[str, Path, Path]]:
    expert_by_id: dict[str, Path] = {}
    for path in list_volumes(experts):
        cid = case_id_from_name(path.name)
        if cid in expert_by_id:
            raise ValueError(f"duplicate expert case id {cid}: {path}")
        expert_by_id[cid] = path
    pairs: list[tuple[str, Path, Path]] = []
    missing: list[str] = []
    seen: list[str] = []
    for path in list_volumes(scans):
        cid = case_id_from_name(path.name)
        seen.append(cid)
        expert = expert_by_id.get(cid)
        if expert is None:
            missing.append(cid)
        else:
            pairs.append((cid, path, expert))
    extra = sorted(set(expert_by_id) - set(seen))
    if missing or extra:
        raise ValueError(f"unpaired in {scans.parent}: missing={missing} extra={extra}")
    return pairs


def find_splits(src_root: Path) -> dict[str, tuple[Path, Path]]:
    found: dict[str, tuple[Path, Path]] = {}
    for scans in src_root.rglob("micro_ultrasound_scans"):
        if not scans.is_dir():
            continue
        expert = scans.parent / "expert_annotations"
        split = scans.parent.name
        if split not in {"train", "test"} or not expert.is_dir():
            continue
        if split in found:
            raise ValueError(f"more than one {split} tree under {src_root}")
        found[split] = (scans, expert)
    if "train" not in found or "test" not in found:
        raise FileNotFoundError(
            "Need both train/ and test/, each with micro_ultrasound_scans/ "
            f"and expert_annotations/. Looked under {src_root}."
        )
    return found


def ignored_annotation_files(src_root: Path) -> int:
    keep = {"micro_ultrasound_scans", "expert_annotations"}
    count = 0
    for folder in src_root.rglob("*"):
        if not folder.is_dir() or folder.name in keep:
            continue
        if folder.parent.name not in {"train", "test"}:
            continue
        count += len(list_volumes(folder))
    return count


def write_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.name.endswith(".nii.gz"):
        shutil.copy2(src, dst)
        return
    image = sitk.ReadImage(str(src))
    sitk.WriteImage(image, str(dst), useCompression=True)


def write_expert_label(image_path: Path, label_path: Path, dst: Path) -> tuple[float, float, float]:
    image = sitk.ReadImage(str(image_path))
    label = sitk.ReadImage(str(label_path))
    if image.GetSize() != label.GetSize():
        raise ValueError(
            f"size mismatch {image_path.name} {image.GetSize()} vs {label_path.name} {label.GetSize()}"
        )
    if not np.allclose(image.GetSpacing(), label.GetSpacing(), rtol=0, atol=1e-5):
        raise ValueError(
            f"spacing mismatch {image_path.name} {image.GetSpacing()} vs {label.GetSpacing()}"
        )
    binary = (sitk.GetArrayFromImage(label) > 0).astype(np.uint8)
    if binary.max() != 1 or binary.min() != 0:
        raise ValueError(f"{label_path.name} has no foreground or is empty after binarizing")
    out = sitk.GetImageFromArray(binary)
    out.CopyInformation(image)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(dst), useCompression=True)
    spacing = image.GetSpacing()
    return (float(spacing[0]), float(spacing[1]), float(spacing[2]))


def spacing_needs_review(spacing: tuple[float, float, float]) -> bool:
    return any(v <= 0 for v in spacing) or all(abs(v - 1.0) < 1e-6 for v in spacing)


def convert(src_root: Path, raw_root: Path, write: bool) -> None:
    splits = find_splits(src_root)
    train_pairs = pair_split(*splits["train"])
    test_pairs = pair_split(*splits["test"])
    ignored = ignored_annotation_files(src_root)
    dataset_dir = raw_root / DATASET_NAME
    heldout = raw_root / "labelsTs_expert_heldout"

    print(f"source {src_root}")
    print(f"train cases {len(train_pairs)} (Zenodo description says 55)")
    print(f"test cases {len(test_pairs)} (Zenodo description says 20)")
    print(f"other annotation files ignored {ignored}")
    print(f"dataset {dataset_dir}")
    if len(train_pairs) != 55 or len(test_pairs) != 20:
        print("WARNING: case counts differ from 55 train / 20 test. Pairing still succeeded.")

    review: list[str] = []
    if not write:
        for cid, image, expert in train_pairs[:3] + test_pairs[:1]:
            print(f"  {cid}: {image.name} + {expert.name}")
        print("dry run only. Re-run with --write to create the nnU-Net files.")
        return

    if dataset_dir.exists():
        raise FileExistsError(f"{dataset_dir} already exists. Remove it before rewriting.")

    for cid, image, expert in train_pairs:
        write_image(image, dataset_dir / "imagesTr" / f"{cid}_0000.nii.gz")
        spacing = write_expert_label(
            image, expert, dataset_dir / "labelsTr" / f"{cid}.nii.gz"
        )
        if spacing_needs_review(spacing):
            review.append(cid)
        print(f"train {cid} spacing {tuple(round(v, 4) for v in spacing)}")

    for cid, image, expert in test_pairs:
        write_image(image, dataset_dir / "imagesTs" / f"{cid}_0000.nii.gz")
        spacing = write_expert_label(image, expert, heldout / f"{cid}.nii.gz")
        if spacing_needs_review(spacing):
            review.append(cid)
        print(f"test  {cid} spacing {tuple(round(v, 4) for v in spacing)}")

    doc = {
        "channel_names": {"0": "microUS"},
        "labels": {"background": 0, "prostate": 1},
        "numTraining": len(train_pairs),
        "file_ending": ".nii.gz",
        "name": DATASET_NAME,
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {dataset_dir / 'dataset.json'}")
    print(f"held-out test masks {heldout}")
    if review:
        print("spacing to review (placeholder 1,1,1 or a non-positive value):")
        print(" ", " ".join(review))
    else:
        print("no placeholder spacing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "src_root",
        type=Path,
        help="Folder that contains train/ and test/ from the Zenodo zip",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="nnUNet_raw. Default: the nnUNet_raw environment variable",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Create Dataset810. Without this flag, only print the pairing.",
    )
    args = parser.parse_args()
    raw = args.raw
    if raw is None:
        import os

        env = os.environ.get("nnUNet_raw")
        if not env:
            print("nnUNet_raw is unset. Activate nnunet and source ~/.bashrc, or pass --raw.", file=sys.stderr)
            return 1
        raw = Path(env)
    if not args.src_root.is_dir():
        print(f"not a directory: {args.src_root}", file=sys.stderr)
        return 1
    try:
        convert(args.src_root, raw, write=args.write)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
