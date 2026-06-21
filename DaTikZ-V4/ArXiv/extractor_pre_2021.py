from datetime import datetime
from functools import partial
from multiprocessing import Pool
from os import sched_getaffinity
from os.path import basename
from shutil import copy
from tempfile import TemporaryDirectory
from typing import Callable
from tqdm import tqdm
from downloader import delete, download
from cleaner import ArxivCleaner


LATEX_DIR = "ArXiv/arxiv_extracted"
ENVS = [b"tikzpicture", b"circuitikz", b"tikzcd", b"\\usetikzlibrary", b"\\tikz", b"quantikz"]


def clean(archive, output, target_dir=LATEX_DIR, filter_func=lambda _: True, verbose=False):
    with TemporaryDirectory() as work_dir:
        arxiv_cleaner = ArxivCleaner(
            data_dir=archive,
            work_dir=work_dir,
            target_dir=target_dir,
            filter_func=filter_func
        )
        return arxiv_cleaner.run(out_fname=output, verbose=verbose)


def process(archive, **kwargs):
    if isinstance(archive, Callable):
        archive = archive()
    output = f"{basename(archive)}.jsonl"
    path = clean(archive, output, **kwargs)
    delete(archive)
    return path


if __name__ == "__main__":
    # def filter_func(tex): return b"tikzpicture" in tex
    def filter_func(tex):
        return any(env in tex for env in ENVS)
    # cutoff = datetime(2005, 10, 23) # tikz 1.0 release date
    cutoff = datetime(2021, 1, 1)
    with Pool(num_workers:=len(sched_getaffinity(0))) as p:
        with TemporaryDirectory() as target_dir:
            tasks = list(download(lazy=True, cutoff=cutoff))
            kwargs = dict(filter_func=filter_func, target_dir=target_dir)
            for path in tqdm(p.imap_unordered(partial(process, **kwargs), tasks), total=len(tasks)):
                copy(path, LATEX_DIR)