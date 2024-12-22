import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
  """ Trains a PyTorch model on a single epoch

  Args:
    model: A PyTorch model to be trained
    dataloader: A DataLoader instance for the model to be trained on
    loss_fn: A PyTorch loss function to minimize
    optimizer: A PyTorch optimizer to help minimize the loss function
    device: A target device to compute on (e.g. "cuda" or "cpu" or "mps")

  Returns:
    A tuple of training loss and training accuracy metrics
  """

  model.train()
  train_loss, train_acc = 0, 0

  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    train_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Predicting class and accumulating accuracy metrics
    y_pred_labels = torch.argmax(y_pred, dim=1)
    train_acc += (y_pred_labels == y).sum().item() / len(y_pred)

  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
  """ Tests a PyTorch model on a single epoch

  Args:
    model: A PyTorch model to be trained
    dataloader: A DataLoader instance for the model to be trained on
    loss_fn: A PyTorch loss function to minimize
    device: A target device to compute on (e.g. "cuda" or "cpu" or "mps")

  Returns:
    A tuple of test/validation loss and accuracy metrics
  """

  model.eval()
  test_loss, test_acc = 0, 0

  with torch.inference_mode():
    for batch, (X, y) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)
      y_pred = model(X)
      loss = loss_fn(y_pred, y)
      test_loss += loss

      y_pred_labels = torch.argmax(y_pred, dim=1)
      test_acc += (y_pred_labels == y).sum().item() / len(y_pred_labels)

  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device
) -> Dict[str, List]:
  """ Trains and validates PyTorch model

  Args:
    model: A PyTorch model to be trained
    train_dataloader: A DataLoader instance for the model to be trained on
    test_dataloader: A DataLoader instance for the model to be trained on
    loss_fn: A PyTorch loss function to minimize
    optimizer: A PyTorch optimizer to help minimize the loss function
    epochs: An integer indicating how many iterations to train for
    device: A target device to compute on (e.g. "cuda" or "cpu" or "mps")

  Returns:
    A tuple of training loss and training accuracy metrics
  """

  results = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
  }

  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model=model,
                                       dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=device)

    val_loss, val_acc = test_step(model=model,
                                  dataloader=val_dataloader,
                                  loss_fn=loss_fn,
                                  device=device) 

    print(
      f"Epoch: {epoch+1} |"
      f"train_loss: {train_loss:.4f} |"
      f"train_acc: {train_acc:.4f} |"
      f"val_loss: {val_loss:.4f} |"
      f"val_acc: {val_acc:.4f} |"
    )

    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["val_loss"].append(val_loss)
    results["val_ac"].append(val_acc)

  return results


def test(
  model: torch.nn.Module,
  test_dataloader: torch.utils.data.DataLoader,
  loss_fn: torch.nn.Module,
  epochs: int,
  device: torch.device
) -> Dict[str, List]:
  """ Trains and validates PyTorch model

  Args:
    model: A PyTorch model to be trained
    test_dataloader: A DataLoader instance for the model to be tested on
    loss_fn: A PyTorch loss function to minimize
    epochs: An integer indicating how many iterations to train for
    device: A target device to compute on (e.g. "cuda" or "cpu" or "mps")

  Returns:
    A tuple of training loss and training accuracy metrics
  """

  results = {
    "test_loss": [],
    "test_acc": [],
  }

  for epoch in tqdm(range(epochs)):
    test_loss, test_acc = test_step(model=model,
                                  dataloader=test_dataloader,
                                  loss_fn=loss_fn,
                                  device=device) 

    print(
      f"Epoch: {epoch+1} |"
      f"test_loss: {test_loss:.4f} |"
      f"test_acc: {test_acc:.4f} |"
    )

    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  return results