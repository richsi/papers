import torch
from torch.utils.data import DataLoader

def create_dataloaders(
    train_data: torch.utils.data.Dataset,
    val_data: torch.utils.data.Dataset,
    test_data: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int
):
  """Creates training, validation, and testing DataLoaders.
  Args:
      train_data (Dataset): Training dataset.
      val_data (Dataset): Validation dataset.
      test_data (Dataset): Testing dataset.
      batch_size (int): Number of samples per batch.
      num_workers (int): Number of subprocesses for data loading.

  Returns:
      Tuple[DataLoader, DataLoader, DataLoader, List[str]]: 
      - train_loader: DataLoader for training data.
      - val_loader: DataLoader for validation data.
      - test_loader: DataLoader for testing data.
      - class_names: List of class names for the dataset.
    """
  # Creating DataLoaders
  train_loader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True     # Load samples on CPU and pushes to GPU during training (speeds up transfer for big datasets)
  )

  val_loader = DataLoader(
    dataset=val_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True     
  )

  test_loader = DataLoader(
    dataset=val_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True     
  )

  # Extracting class names
  try:
    # Case 1: Dataset has a 'classes' attribute
    class_names = test_data.classes
  except AttributeError:
    # Case 2: Dataset has a 'targets' attribute
    if hasattr(test_data, 'targets'):
      unique_classes = set(test_data.targets)
      class_names = [str(cls) for cls in sorted(unique_classes)]
    # Case 3: Dataset has a 'labels' attribute
    elif hasattr(test_data, 'labels'):
      unique_classes = set(test_data.labels)
      class_names = [str(cls) for cls in sorted(unique_classes)]
    else:
      print("Warning: Could not determine class names. Assigning numerical labels.")
      class_names = [str(i) for i in range(len(set(test_data)))]

  return train_loader, val_loader, test_loader, class_names