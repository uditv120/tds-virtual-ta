import os
import requests
from pathlib import Path

def download_github_repo_markdown(repo_url: str, save_dir: str):
    os.system(f"git clone {repo_url} {save_dir}")

# Example usage
if __name__ == "__main__":
    REPO = "https://github.com/uditv120/tools-in-data-science-public"
    download_github_repo_markdown(REPO, "course_content")
