"""
Re-implementation of

Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385v1

on the CIFAR-10 dataset.
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from argparse import ArgumentParser
from torch.utils.data import random_split
from torchvision import models

from utils.data_setup import *
from src.arch.resnet.model import ResNet

def main(args):
  """
  Train reimplementation ResNet model using CIFAR10 dataset
  """

  # Unpacking arguments
  batch_size = args.batch_size
  epochs = args.epochs
  lr = args.lr
  save_dir = args.save
  seed = args.seed

  torch.manual_seed(seed)

  # Transforming to ResNet size
  transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
  ])

  # Downloading and splitting CIFAR10 dataset
  train_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
  train_size = int(0.8 * len(train_dataset_full))
  val_size = len(train_dataset_full) - train_size

  train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])
  test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

  # Loading the data
  train_loader, val_loader, test_loader, class_names = create_dataloaders(
    train_data=train_dataset,
    val_data=val_dataset,
    test_data=test_dataset,
    batch_size=batch_size,
    num_workers=os.cpu_count()
  )

  # Init the model
  # model = ResNet()

  print(train_dataset[0][0].shape, train_dataset[0][1])
  print(models.resnet34())

if __name__ == "__main__":
  parser = ArgumentParser()

  # Model hyper-parameters

  # Training parameters
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--epochs", type=int, default=5)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--save", type=str, default="checkpoints/resnet")

  parser.add_argument("--seed", type=int, default=42)

  args = parser.parse_args()
  print(args)
  main(args) 