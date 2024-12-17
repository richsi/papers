import os
import requests
from pathlib import Path

def download_data(
  source: str,
  destination: str
):
  """Downloads data into cwd
  Takes in a source url to the data and downloads it into the destination directory.
  Creates directory if destination does not exist.

  Args:
    source: String URL to dataset
    destination: String directory to store dataset in

  Returns: 
    None
  """
  data_path = Path("data/")
  dest_path = data_path / destination
  
  if not dest_path.is_dir():
    dest_path.mkdir(parents=True, exist_ok=True)