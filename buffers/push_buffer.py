# This script provides functions to upload and download buffers (e.g., ReplayBuffer) to/from Hugging Face Hub.
# It assumes the buffer is saved as a file (e.g., .pkl, .pt, .npz, etc.)

from huggingface_hub import HfApi, HfFolder, Repository, upload_file, hf_hub_download, list_repo_files, repo_info
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
    try:
        print(f"Attempting to download {filename} from {repo_id}...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            cache_dir=local_dir,
            token=token,
            revision=revision
        )
        print(f"Successfully downloaded {filename} from {repo_id} to {local_path}")
        return local_path
    except Exception as e:
        print(f"Error downloading {filename} from {repo_id}: {e}")
        print(f"Please check:")
        print(f"  1. Repository exists: https://huggingface.co/{repo_id}")
        print(f"  2. File exists in the repository")
        print(f"  3. You have access to the repository (if private)")
        print(f"  4. Your HuggingFace token is valid (if required)")
        raise

def check_repo_exists(repo_id, token=None):
    """
    Checks if a Hugging Face Hub repository exists.

    Args:
        repo_id (str): Hugging Face repo id, e.g. "username/repo_name".
        token (str, optional): Hugging Face token. If None, uses default.
    Returns:
        bool: True if repository exists, False otherwise.
    """
    try:
        info = repo_info(repo_id=repo_id, repo_type="model", token=token)
        print(f"Repository {repo_id} exists and is accessible.")
        return True
    except Exception as e:
        print(f"Repository {repo_id} does not exist or is not accessible: {e}")
        return False

def list_repo_contents(repo_id, token=None, revision="main"):
    """
    Lists all files in a Hugging Face Hub repository.

    Args:
        repo_id (str): Hugging Face repo id, e.g. "username/repo_name".
        token (str, optional): Hugging Face token. If None, uses default.
        revision (str): Branch or commit to list from.
    """
    try:
        print(f"Listing files in {repo_id}...")
        files = list_repo_files(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            revision=revision
        )
        if files:
            print(f"Found {len(files)} files:")
            for file in files:
                print(f"  - {file}")
        else:
            print("No files found in the repository.")
        return files
    except Exception as e:
        print(f"Error listing files in {repo_id}: {e}")
        raise

import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Buffer HuggingFace Hub utilities")
    parser.add_argument("--download", action="store_true", help="Download buffer from HuggingFace Hub")
    parser.add_argument("--upload", action="store_true", help="Upload buffer to HuggingFace Hub")
    parser.add_argument("--list", action="store_true", help="List files in HuggingFace Hub repository")
    parser.add_argument("--check", action="store_true", help="Check if HuggingFace Hub repository exists")
    parser.add_argument("--repo_id", type=str, default="korneelf1/neurips", help="HuggingFace repo id")
    parser.add_argument("--filename", type=str, default="l2f_buffer_1996.hdf5", help="Filename in the repo")
    parser.add_argument("--local_dir", type=str, default=".", help="Local directory to save the buffer")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token")
    parser.add_argument("--revision", type=str, default="main", help="Repo revision (branch/commit)")

    args = parser.parse_args()

    if args.download:
        print("Downloading buffer from HuggingFace Hub...")
        download_buffer_from_hf(
            repo_id=args.repo_id,
            filename=args.filename,
            local_dir=args.local_dir,
            token=args.token,
            revision=args.revision
        )
    elif args.upload:
        print("Uploading buffer to HuggingFace Hub...")
        # Check if the file exists before uploading
        if not os.path.exists(args.filename):
            print(f"Error: File {args.filename} does not exist!")
            sys.exit(1)
        upload_buffer_to_hf(
            buffer_file_path=args.filename,
            repo_id=args.repo_id,
            token=args.token
        )
    elif args.list:
        print("Listing files in HuggingFace Hub repository...")
        list_repo_contents(
            repo_id=args.repo_id,
            token=args.token,
            revision=args.revision
        )
    elif args.check:
        print("Checking if HuggingFace Hub repository exists...")
        check_repo_exists(
            repo_id=args.repo_id,
            token=args.token
        )
    else:
        print("Please specify one of: --download, --upload, --list, or --check")
        parser.print_help()
