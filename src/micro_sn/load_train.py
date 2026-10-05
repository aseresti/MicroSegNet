from pathlib import Path
import numpy as np
import SimpleITK as sitk

def case_id(path):
    stem = Path(path).name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    elif stem.endswith(".nii"):
        stem = stem[:-4]
    split_name, number = stem.split("_")[-2:]
    return f"{split_name}_{number}"

def load_case(image_path, expert_path, non_expert_path):
    img = sitk.ReadImage(str(image_path))
    gt = sitk.ReadImage(str(expert_path))
    st = sitk.ReadImage(str(non_expert_path))
    assert img.GetSize() == gt.GetSize() == st.GetSize(), "Image and masks must have the same size"
    assert case_id(image_path) == case_id(expert_path) == case_id(non_expert_path), (
        case_id(image_path), case_id(expert_path), case_id(non_expert_path)
    )
    image = sitk.GetArrayFromImage(img).astype(np.float32)
    expert = sitk.GetArrayFromImage(gt)
    non_expert = sitk.GetArrayFromImage(st)

    image = image / 255.0  # Normalize image to [0, 1]
    expert = (expert > 0).astype(np.uint8)  # Convert to binary
    non_expert = (non_expert > 0).astype(np.uint8)
    return image, expert, non_expert


