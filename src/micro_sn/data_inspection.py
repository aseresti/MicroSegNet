import sys

from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch

root = Path(__file__).parents[2]
path_to_data = root / "data" / "Micro_Ultrasound_Prostate_Segmentation_Dataset"

sys.path.insert(0, str(root / "src" ))

from micro_sn.load_train import case_id, load_case, frame_to_tensor
from micro_sn.loss import downsample_mask, ag_bce, multiscale_ag_bce
from micro_sn.model import FourScaleStub

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

print("\n\nDownsampling the mask:")
gt_112, st_112 = downsample_mask(gt_tensor, 112), downsample_mask(st_tensor, 112)
gt_56, st_56 = downsample_mask(gt_tensor, 56), downsample_mask(st_tensor, 56)
gt_28, st_28 = downsample_mask(gt_tensor, 28), downsample_mask(st_tensor, 28)
for name, gt, st in [("224", gt_tensor, st_tensor), ("112", gt_112, st_112), ("56", gt_56, st_56), ("28", gt_28, st_28)]:
    native_fraction = float(expert[z].sum()) / expert[z].size
    tensor_fraction = float(gt.sum()) / gt.numel()
    hard_pixels = int((gt != st).sum())
    print(f"Size: {name}, GT shape: {gt.shape}, ST shape: {st.shape}, Native fraction: {native_fraction:.4f}, Tensor fraction: {tensor_fraction:.4f}, Hard pixels: {hard_pixels}")

# computing the ag_bce loss using the ag_bce function
print("\n\nComputing the ag_bce loss:")
expert_ = torch.tensor([[1, 1], [0, 0]])
student_ = torch.tensor([[1, 0], [0, 0]])
pred = torch.full((2, 2), 0.5)
loss = ag_bce(pred, expert_, student_)
term = -torch.log(pred[0, 0] + 1e-6) # raw bce at p=0.5
expected = (3 * 1 + 1 * 4) * term / 4 # three easy pixels with weight 1, one hard pixel with weight 4, divided by total weight
print(f"Loss: {float(loss):.4f}, Expected: {float(expected):.4f}, Ratio: {float(loss / expected):.4f}")
assert torch.allclose(loss, expected), "ag_bce loss does not match expected value"

# computing the multiscale_ag_bce loss using the multiscale_ag_bce function
print("\n\nComputing the multiscale_ag_bce loss:")
pairs = [(gt_tensor, st_tensor), (gt_112, st_112), (gt_56, st_56), (gt_28, st_28)]
preds = [torch.full_like(gt, 0.5, dtype=torch.float32) for gt, _ in pairs]
loss = multiscale_ag_bce(preds, [gt for gt, _ in pairs], [st for _, st in pairs])

term = -torch.log(torch.tensor(0.5 + 1e-6)) # raw bce at p=0.5
expected = torch.zeros(())
for gt, st in pairs:
    hard_fraction = (gt != st).float().mean()
    scale_loss = term * (1 + 3 * hard_fraction)
    expected = expected + scale_loss
    print(tuple(gt.shape), "hard fraction", round(float(hard_fraction), 4), "scale loss", round(float(scale_loss), 4))

print("sum of scale losses:", round(float(expected), 4))
assert torch.allclose(loss, expected), "multiscale_ag_bce loss does not match expected value"

# testing the FourScaleStub model
print("\n\nTesting the FourScaleStub model:")
model = FourScaleStub()
x = img_tensor.unsqueeze(0)  # Add batch dimension
logits = model(x)
probs = [torch.sigmoid(t.squeeze(1)) for t in logits]  # Remove channel dimension and apply sigmoid
gt_tensor, st_tensor = gt_tensor.squeeze(0), st_tensor.squeeze(0)  # Add batch dimension
experts = [m.unsqueeze(0) for m in [gt_tensor, gt_112, gt_56, gt_28]]  # Remove batch dimension
students = [m.unsqueeze(0) for m in [st_tensor, st_112, st_56, st_28]]  # Remove batch dimension

loss = multiscale_ag_bce(probs, experts, students)
loss.backward()

for p in probs:
    print(f"Probabilities shape: {p.shape}, min: {p.min().item():.4f}, max: {p.max().item():.4f}")
print(f"Multiscale loss: {loss.item():.4f}")
print("grad", model.conv.weight.grad.abs().mean().item())