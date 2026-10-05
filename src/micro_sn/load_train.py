from pathlib import Path
import numpy as np
import SimpleITK as sitk
import cv2
import torch

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

def frame_to_tensor(image, expert, nonexpert, z, size=224):
    img = cv2.resize(image[z], (size, size), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(expert[z], (size, size), interpolation=cv2.INTER_NEAREST)
    st = cv2.resize(nonexpert[z], (size, size), interpolation=cv2.INTER_NEAREST)
    img_tensor = torch.from_numpy(img).unsqueeze(0).float()  # Add channel dimension
    gt_tensor = torch.from_numpy(gt).unsqueeze(0).long()
    st_tensor = torch.from_numpy(st).unsqueeze(0).long()
    return img_tensor, gt_tensor, st_tensor



