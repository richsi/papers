import torch
import torch.nn as nn

class BasicBlock(nn.Module):
  """
  Building residual block
  """
  def __init__(self, in_size, out_size, stride=1, kernel_size=3, downsample=None):
    super().__init__()

    self.conv1 = nn.Conv2d(in_size, out_size, kernel_size, stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_size, eps=1e-05, momentum=0.1, affine=True)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_size, eps=1e-05, momentum=0.1, affine=True)

    self.downsample = downsample

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    # print(identity.shape, " ", out.shape)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)
    return out


class ResNet(nn.Module):
  """
  Assembling ResNet architecture
  """
  def __init__(self, block, layers, num_classes=10):
    super().__init__()

    self.in_size = 64
    self.block0 = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    # res_blocks = [3, 4, 6, 3]
    self.block1 = self._make_layer(block, out_size=64, blocks=layers[0])
    self.block2 = self._make_layer(block, out_size=128, blocks=layers[1], stride=2)
    self.block3 = self._make_layer(block, out_size=256, blocks=layers[2], stride=2)
    self.block4 = self._make_layer(block, out_size=512, blocks=layers[3], stride=2)

    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(512, num_classes)

  def _make_layer(self, block, out_size, blocks, stride=1):
    downsample = None
    if stride != 1 or self.in_size != out_size:
      downsample = nn.Sequential(
        nn.Conv2d(self.in_size, out_size, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_size)
      )
    # Creating the layers of residual connection blocks
    layers = []
    layers.append(block(self.in_size, out_size, stride, downsample=downsample))
    self.in_size = out_size
    for _ in range(1, blocks):
      layers.append(block(out_size, out_size))
    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.block0(x)
    out = self.block1(out)
    out = self.block2(out)
    out = self.block3(out)
    out = self.block4(out)
    out = self.avgpool(out)
    out = torch.flatten(out, 1)
    out = self.fc(out)
    return out


def resnet34():
  layers = [3, 4, 6, 3]
  model = ResNet(BasicBlock, layers)
  return model