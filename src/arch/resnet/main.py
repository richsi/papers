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

from modules import engine, utils
from src.arch.resnet.model import resnet34

def main(args):
  """
  Train reimplementation ResNet model using CIFAR10 dataset
  """

  # Unpacking arguments
  batch_size = args.batch_size
  epochs = args.epochs
  lr = args.lr
  momentum = args.momentum
  decay = args.decay
  save_dir = args.save_dir
  seed = args.seed
  name = args.name

  device = "cuda" if torch.cuda.is_available() else "mps"

  torch.manual_seed(seed)

  # Transforming to ResNet size
  transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  # Downloading and splitting CIFAR10 dataset
  train_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
  train_size = int(0.8 * len(train_dataset_full))
  val_size = len(train_dataset_full) - train_size

  train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])
  test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

  # Loading the data
  train_loader, val_loader, test_loader, class_names = utils.create_dataloaders(
    train_data=train_dataset,
    val_data=val_dataset,
    test_data=test_dataset,
    batch_size=batch_size,
    num_workers=os.cpu_count()
  )

  model = resnet34()
  model = model.to(device)

  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(params=model.parameters(),
                              lr=lr,
                              momentum=momentum,
                              weight_decay=decay)
  
  results = engine.train(model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=epochs,
        device=device)

  utils.save_model(model=model,
                   save_dir=save_dir,
                   model_name=f"{name}.pth")

  utils.plot_results(results, name)

if __name__ == "__main__":
  parser = ArgumentParser()

  # Training parameters
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--epochs", type=int, default=5)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--momentum", type=float, default=0.9)
  parser.add_argument("--decay", type=float, default=1e-4)
  parser.add_argument("--save_dir", type=str, default="models")
  parser.add_argument("--name", type=str, default="resnet34")

  parser.add_argument("--seed", type=int, default=42)

  args = parser.parse_args()
  print(args)
  main(args) 