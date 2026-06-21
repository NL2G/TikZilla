import uuid
import fnmatch
import gzip
import json
import logging
import lzma
import tarfile
import os
import pathlib
import re
import concurrent.futures

from datetime import datetime
from tqdm import tqdm
from os.path import basename, dirname
from subprocess import CalledProcessError, DEVNULL, run
from tempfile import NamedTemporaryFile, TemporaryDirectory


ARXIV_URL = "https://arxiv.org/abs/"


class ArxivCleaner:
    def __init__(self, data_dir, work_dir, target_dir, worker_id=None, filter_func=lambda _: True):
        self._data_dir = pathlib.Path(data_dir)
        self._work_dir = pathlib.Path(work_dir)
        self._target_dir = pathlib.Path(target_dir)
        self._worker_id = worker_id if worker_id else str(uuid.uuid4())
        self.filter_func = filter_func
        for d in [self._work_dir, self._target_dir]:
            if not d.exists():
                d.mkdir(parents=True)

    def run_parallel(self, max_files=-1, workers=None, tar_fp_list=None, compress=False):
        out_file = self._target_dir / (f"arxiv_{self._worker_id}.jsonl" + (".xz" if compress else ""))
        with open(out_file, "wb") as f:
            with concurrent.futures.ProcessPoolExecutor(workers) as executor:
                for record, arxiv_id in executor.map(
                        create_record_single_arg,
                        tqdm(self.arxiv_iterator(max_files=max_files, tar_fp_list=tar_fp_list))
                ):
                    if record is None:
                        logging.error(f"failed  to process {arxiv_id}")
                        continue
                    if len(record["text"]) == 0:
                        logging.warning(f"empty text for {arxiv_id}")
                        continue
                    if compress:
                        f.write(lzma.compress((json.dumps(record) + "\n").encode()))
                    else:
                        f.write((json.dumps(record) + "\n").encode())
                    logging.info(f"processed {arxiv_id}")
                executor.shutdown(wait=True)

    def run(self, max_files=-1, out_fname="arxiv.jsonl", compress=False):
        with open((path := self._target_dir / (out_fname + (".xz" if compress else ""))), "wb") as f:
            for tex_file, yymm, arxiv_id, timestamp in tqdm(self.arxiv_iterator(max_files=max_files)):
                record, arxiv_id = create_record(
                    tex_file=tex_file,
                    yymm=yymm,
                    arxiv_id=arxiv_id,
                    timestamp=timestamp
                )
                if record is None:
                    logging.error(f"failed to process {arxiv_id}")
                    continue
                if len(record["text"]) == 0:
                    logging.warning(f"empty text for {arxiv_id}")
                    continue
                if compress:
                    f.write(lzma.compress((json.dumps(record) + "\n").encode()))
                else:
                    f.write((json.dumps(record) + "\n").encode())
                logging.info(f"processed {arxiv_id}")
            return path

    def arxiv_iterator(self, max_files=-1, tar_fp_list=None):
        def _tar_fp_iterator():
            for _tar_fp in tar_fp_list or self._data_dir.glob("*.tar"):
                yield _tar_fp
        failed = 0
        processed = 0
        for tar_fp in _tar_fp_iterator():
            logging.info("start processing {tar_fp}")
            with TemporaryDirectory(dir=self._work_dir) as tmpdir:
                with tarfile.open(tar_fp) as tf:
                    tf.extractall(members=tf.getmembers(), path=tmpdir)
                    for proj_dir_or_file in pathlib.Path(tmpdir).rglob("*.gz"):
                        yymm = proj_dir_or_file.parent.stem
                        arxiv_id = proj_dir_or_file.stem
                        data = _tex_proj_loader(proj_dir_or_file, self.filter_func)
                        if data is None:
                            failed += 1
                            continue
                        tex_file, timestamp = data
                        processed += 1
                        if processed > max_files > 0:
                            break
                        yield tex_file, yymm, arxiv_id, timestamp
                    else:
                        continue
                    break
        logging.info("Failed loading : {failed}")
        logging.info("done.")


def latexpand(tex_file_path):
    with NamedTemporaryFile(buffering=0) as tmp:
        path, file = dirname(tex_file_path) or None, basename(tex_file_path)
        cmd = ["latexpand", "--keep-comments", file, "--output", tmp.name]
        run(cmd, cwd=path, stdout=DEVNULL, stderr=DEVNULL, check=True)
        tmp.seek(0)
        return tmp.read().strip()


def latexpand_str(latex):
    with NamedTemporaryFile(buffering=0) as tmp:
        tmp.write(latex)
        return latexpand(tmp.name)


def find_root_file(directory="."):
    first_file = None
    for root, _, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.tex'):
            path = os.path.join(root, filename)
            if not first_file:
                first_file = path
            with open(path, 'rb') as file:
                content = file.read()
                if any(pattern in content for pattern in [rb'\documentclass', rb'\documentstyle']):
                    return path
    if first_file:
        return first_file
    raise FileNotFoundError


def format_arxiv_id(arxiv_id):
    match = re.search(r'^([a-zA-Z-]*)([\d\.]+)$', arxiv_id)
    if match is None:
        raise ValueError(f"Invalid arxiv id: {arxiv_id}")
    if match.group(1) == "":
        return match.group(2)
    return f"{match.group(1)}/{match.group(2)}"


def create_record_single_arg(args):
    return create_record(*args)


def create_record(tex_file, yymm, arxiv_id, timestamp):
    if len(tex_file) == 0:
        return {"text": "", "meta": {}}, arxiv_id
    try:
        clean_arxiv_id = format_arxiv_id(arxiv_id)
    except Exception as e:
        logging.warning(f"failed to format arxiv id {arxiv_id}; excpetion={e}")
        clean_arxiv_id = arxiv_id
    if timestamp is not None:
        timestamp = datetime.fromtimestamp(timestamp).isoformat()
    return (
        {
            "text": tex_file,
            "meta": {
                "timestamp": timestamp,
                "yymm": yymm,
                "arxiv_id": clean_arxiv_id,
                "url": f"{ARXIV_URL}{clean_arxiv_id}",
                "source": "arxiv"
            }
        },
        clean_arxiv_id
    )


def matches(directory, filter_func):
    for root, _, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.tex'):
            path = os.path.join(root, filename)
            with open(path, 'rb') as file:
                if filter_func(file.read()):
                    return True
    return False


def _tex_proj_loader(file_or_dir_path, filter_func=lambda _: True):
    timestamp = file_or_dir_path.lstat().st_mtime
    try:
        with TemporaryDirectory() as tmpdir:
            with tarfile.open(file_or_dir_path, "r") as sub_tf:
                sub_tf.extractall(path=tmpdir)
                try:
                    if matches(tmpdir, filter_func):
                        file_content = latexpand(find_root_file(tmpdir))
                    else:
                        return None
                except (FileNotFoundError, CalledProcessError) as e:
                    logging.error(f"{type(e).__name__}: {file_or_dir_path}")
                    return None
    except tarfile.ReadError:
        try:
            with gzip.open(file_or_dir_path, "rb") as gz:
                file_content = latexpand_str(gz.read())
        except Exception as e:
            logging.error(f"{type(e).__name__}: {file_or_dir_path}")
            return None
    except Exception as e:
        logging.error(f"{type(e).__name__}: {file_or_dir_path}")
        return None
    for idx, encoding in enumerate(encodings:=["utf-8", "latin1"]):
        try:
            return file_content.decode(encoding), timestamp
        except ValueError:
            if idx == len(encodings) -1:
                logging.error(f"DecodeError: {file_or_dir_path}")