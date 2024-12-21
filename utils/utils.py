import torch
from pathlib import Path

def save_model(
    model: torch.nn.Module,
    target_dir: str,
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
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt")
  model_save_path = target_dir_path / model_name

  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(), f=model_save_path)