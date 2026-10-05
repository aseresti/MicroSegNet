import torch.nn as nn
import torch.nn.functional as F

class FourScaleStub(nn.Module):
    """a temporary stub for a four-scale model to run loss.backward() in data_inspection.py without needing the full model."""
    def __init__(self):
        super(FourScaleStub, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        base = self.conv(x)
        return [F.interpolate(base, size=s, mode="bilinear", align_corners=False) for s in [224, 112, 56, 28]]
