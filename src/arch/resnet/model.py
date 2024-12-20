import torch
import torch.nn as nn

class BasicBlock(nn.Module):
  """
  Building residual block
  """
  def __init__(self, in_size, out_size, stride, padding, kernel_size=3):
    super().__init__()

    
    self.conv1 = nn.Conv2d(in_size, out_size, kernel_size, stride, padding, bias=False)
    self.bn1 = nn.BatchNorm2d(out_size, eps=1e-05, momentum=0.1, affine=True)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, stride, padding, bias=False)
    self.bn2 = nn.BatchNorm2d(out_size, eps=1e-05, momentum=0.1, affine=True)

    def forward(self, x):
      identity = x

      out = self.conv1(x)
      out = self.bn1(out)
      out = self.relu(out)
      out = self.conv2(out)
      out = self.bn2(out)

      out += identity
      out = self.relu(out)
      return out


class ResNet(nn.Module):
  """
  Assembling ResNet architecture
  """
  def __init__(self, block):
    super().__init__()

    self.block0 = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    self.layer1 = None

    def forward(self):
      pass

    def train_step(self):
      pass

    def val_step(self):
      pass

    def test_step(self):
      pass