import sys

from pathlib import Path
import numpy as np
import SimpleITK as sitk

root = Path(__file__).parents[2]
path_to_data = root / "data" / "Micro_Ultrasound_Prostate_Segmentation_Dataset"

sys.path.insert(0, str(root / "src" ))

from micro_sn.load_train import case_id, load_case, frame_to_tensor

def niftis(folder):
    return sorted(Path(folder).glob("*.nii.gz"))

img_paths = niftis(path_to_data / "train" / "micro_ultrasound_scans")
gt_paths = niftis(path_to_data / "train" / "expert_annotations")
st_paths = niftis(path_to_data / "train" / "non_expert_annotations")
test_img = niftis(path_to_data / "test" / "micro_ultrasound_scans")
test_gt = niftis(path_to_data / "test" / "expert_annotations")

# printing the number of images and masks in the training and test sets
print(f"Number of training images: {len(img_paths)}")
print(f"Number of training ground truth: {len(gt_paths)}")
print(f"Number of training student segmentation masks: {len(st_paths)}")
print(f"Number of test images: {len(test_img)}")
print(f"Number of test ground truth: {len(test_gt)}")

img = sitk.ReadImage(str(img_paths[0]))
gt = sitk.ReadImage(str(gt_paths[0]))
st = sitk.ReadImage(str(st_paths[0]))
image = sitk.GetArrayFromImage(img)
expert = sitk.GetArrayFromImage(gt)
student = sitk.GetArrayFromImage(st)

# printing the shape, dtype, min and max values of the image and masks
print("\n\nLoading the first case:")
print(f"Image: {image.shape}, {image.dtype}, min: {np.min(image)}, max: {np.max(image)}")
print(f"Expert: {expert.shape}, {expert.dtype}, min: {np.min(expert)}, max: {np.max(expert)}")
print(f"Student: {student.shape}, {student.dtype}, min: {np.min(student)}, max: {np.max(student)}")
print("spacing:", img.GetSpacing(), "size xyz:", img.GetSize())

z = image.shape[0] // 2 # taking the slice in the middle of the volume
hard = (expert[z] > 0) != (student[z] > 0)
print("mid slice:", z, "hard pixels:", np.sum(hard), "of", hard.size)

# loading a case using the load_case function
print("\n\nLoading a case:")
image, expert, student = load_case(img_paths[0], gt_paths[0], st_paths[0])
print(f"Loaded case: {case_id(img_paths[0])}, image shape: {image.shape}, expert shape: {expert.shape}, student shape: {student.shape}")
print(img_paths[0].name, gt_paths[0].name, st_paths[0].name)

# converting a frame to tensor using the frame_to_tensor function
print("\n\nConverting a frame to tensor:")
img_tensor, gt_tensor, st_tensor = frame_to_tensor(image, expert, student, z)
print(img_tensor.shape,img_tensor.dtype, img_tensor.min(), img_tensor.max())
print(gt_tensor.shape,gt_tensor.dtype, gt_tensor.min(), gt_tensor.max())
print(st_tensor.shape,st_tensor.dtype, st_tensor.min(), st_tensor.max())

native_fraction = float(expert[z].sum()) / expert[z].size
tensor_fraction = float(gt_tensor.sum()) / gt_tensor.numel()
print(f"Native fraction of positive pixels: {native_fraction:.4f}, Tensor fraction of positive pixels at 224: {tensor_fraction:.4f}")
print("hard pixels", int((gt_tensor != st_tensor).sum()))