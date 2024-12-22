import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


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



def save_model(
    model: torch.nn.Module,
    save_dir: str,
    model_name: str
):
  """ Saves a PyTorch model to a target directory

  Args:
    model: Target PyTorch model to save
    target_dir: Directory for saving the model to
    model_name: Filename for the saved model. Should include ".pth" or ".pt" as file extension

  Example usage:
    save_model(model=model, target_dir="models", model_name="01_model_name.pth")
  """

  # Create target directory
  target_dir_path = Path(save_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt")
  model_save_path = target_dir_path / model_name

  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(), f=model_save_path)



def plot_results(results, model_name):
    # Ensure the output directory exists
    os.makedirs("plots", exist_ok=True)

    # Create a figure and the primary y-axis for Loss
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title(f"{model_name} Training and Validation Loss and Accuracy")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color="blue")

    # Plot Loss (Train and Val)
    ax1.plot(results["train_loss"], label="Train Loss", color="blue", linestyle="--", marker="o")
    ax1.plot(results["val_loss"], label="Validation Loss", color="cyan", linestyle="--", marker="o")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.legend(loc="upper left")

    # Create a secondary y-axis for Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="green")

    # Plot Accuracy (Train and Val)
    ax2.plot(results["train_acc"], label="Train Accuracy", color="green", linestyle="-", marker="x")
    ax2.plot(results["val_acc"], label="Validation Accuracy", color="orange", linestyle="-", marker="x")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.legend(loc="upper right")

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_metrics.png")
    plt.show()

