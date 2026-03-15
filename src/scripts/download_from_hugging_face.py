import os
from huggingface_hub import HfApi, hf_hub_download


def download_file_from_hf(repo_id, file_name, local_dir, repo_type="dataset", revision="main"):
    """
    Download a file from a HuggingFace repository.

    Args:
        repo_id (str): HF repo id (e.g. "username/dataset-name")
        file_name (str): file to download
        local_dir (str): local directory where file will be saved
        repo_type (str): "dataset" or "model"
        revision (str): branch/tag (default: main)

    Returns:
        str: local path of downloaded file
    """

    api = HfApi()

    os.makedirs(local_dir, exist_ok=True)

    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=revision)
    print(f"Total files in repo: {len(files)}")

    if file_name not in files:
        raise ValueError(
            f"File '{file_name}' not found in repo '{repo_id}'.\nAvailable files:\n{files}"
        )

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=file_name,
        repo_type=repo_type,
        local_dir=local_dir
    )

    print(f"Downloaded: {file_name}")
    print(f"Saved to: {local_path}")
