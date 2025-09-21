# This script provides functions to upload and download buffers (e.g., ReplayBuffer) to/from Hugging Face Hub.
# It assumes the buffer is saved as a file (e.g., .pkl, .pt, .npz, etc.)

from huggingface_hub import HfApi, HfFolder, Repository, upload_file, hf_hub_download
import os

def upload_buffer_to_hf(buffer_file_path, repo_id, path_in_repo=None, token=None, commit_message="Upload buffer"):
    """
    Uploads a buffer file to a Hugging Face Hub repository.

    Args:
        buffer_file_path (str): Local path to the buffer file.
        repo_id (str): Hugging Face repo id, e.g. "username/repo_name".
        path_in_repo (str, optional): Path in the repo to store the file. Defaults to filename.
        token (str, optional): Hugging Face token. If None, uses default.
        commit_message (str): Commit message for the upload.
    """
    if path_in_repo is None:
        path_in_repo = os.path.basename(buffer_file_path)
    upload_file(
        path_or_fileobj=buffer_file_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message=commit_message
    )
    print(f"Uploaded {buffer_file_path} to {repo_id}/{path_in_repo}")


def download_buffer_from_hf(repo_id = "korneelf1/neurips", filename = "l2f_buffer_1996.hdf5", local_dir=".", token=None, revision="main"):
    """
    Downloads a buffer file from a Hugging Face Hub repository.

    Args:
        repo_id (str): Hugging Face repo id, e.g. "username/repo_name".
        filename (str): Name of the file in the repo.
        local_dir (str): Directory to save the downloaded file.
        token (str, optional): Hugging Face token. If None, uses default.
        revision (str): Branch or commit to download from.
    Returns:
        str: Local path to the downloaded file.
    """
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        cache_dir=local_dir,
        token=token,
        revision=revision
    )
    print(f"Downloaded {filename} from {repo_id} to {local_path}")
    return local_path

# Example usage:
upload_buffer_to_hf("my_buffer.pkl", "my-username/my-buffer-repo")
# local_path = download_buffer_from_hf("my-username/my-buffer-repo", "my_buffer.pkl")
