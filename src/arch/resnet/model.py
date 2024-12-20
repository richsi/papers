import torch
import torch.nn as nn

class ResNet(nn.Module):
  """Creates ResNet architecture
  Args:
    
  """
  def __init__(self, block):
    super().__init__()

    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True)
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.block1 = nn.Sequential()
    self.block2 = nn.Sequential()
    self.block3 = nn.Sequential()
    self.block4 = nn.Sequential()

    def forward(self):
      pass

    def train_step(self):
      pass

    def val_step(self):
      pass

    def test_step(self):
      pass