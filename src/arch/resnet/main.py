"""
Re-implementation of

Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385v1

on the ImageNet dataset.
"""

import os
import torch
from argparse import ArgumentParser
from datasets import load_dataset

from src.arch.resnet.model import ResNet

if __name__ == "__main__":
  # parser = ArgumentParser()

  # # Model hyper-parameters

  # # Training parameters
  # parser.add_argument("--batch_size", type=int, default=32)
  # parser.add_argument("--epochs", type=int, default=5)
  # parser.add_argument("--lr", type=float, default=1e-4)
  # parser.add_argument("--save", type=str, default="checkpoints/resnet")

  print("test")