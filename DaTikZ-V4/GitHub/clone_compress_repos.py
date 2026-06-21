import os
import json
import subprocess
import tarfile
import shutil

from concurrent.futures import ThreadPoolExecutor, as_completed


SEEN_REPOS_FILE = "GitHub/not_processed.json"
CLONE_DIR = "GitHub/repos"
DELETE_AFTER_COMPRESS = True
MAX_WORKERS = 4
TIMEOUT = 90
NUM_RETRIES = 3


def load_seen_repos(filepath=SEEN_REPOS_FILE):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def clone_repo(repo_fullname, dest_path):
    repo_url = f"https://github.com/{repo_fullname}.git"
    for _ in range(NUM_RETRIES):
        try:
            subprocess.run(["git", "clone", "--depth", "1", repo_url, dest_path],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           check=True,
                           timeout=TIMEOUT)
            return True
        except subprocess.CalledProcessError:
            continue
        except subprocess.TimeoutExpired:
            continue
    return False


def compress_repo(repo_path):
    tar_path = f"{repo_path}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(repo_path, arcname=os.path.basename(repo_path))
    return tar_path


def process_repo(repo_fullname):
    repo_dirname = repo_fullname.replace("/", "__")
    dest_path = os.path.join(CLONE_DIR, repo_dirname)
    tar_path = f"{dest_path}.tar.gz"
    if os.path.exists(tar_path):
        return
    if not os.path.exists(dest_path):
        success = clone_repo(repo_fullname, dest_path)
        if not success:
            return
    compress_repo(dest_path)
    if DELETE_AFTER_COMPRESS:
        shutil.rmtree(dest_path)
    return


if __name__ == "__main__":
    os.makedirs(CLONE_DIR, exist_ok=True)
    seen_repos = load_seen_repos()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_repo = {executor.submit(process_repo, repo): repo for repo in seen_repos}
        for future in as_completed(future_to_repo):
            _ = future.result()