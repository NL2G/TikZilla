import os
import io
import re
import tarfile
import time
import tempfile
import subprocess
import json
import requests
import unicodedata
import sys
import datetime
import random

from datetime import time as dtime
from concurrent.futures import ThreadPoolExecutor, as_completed
from subprocess import DEVNULL
from tempfile import NamedTemporaryFile
from lxml import etree
from tqdm import tqdm
from pathlib import Path


NAMESPACES = {
    'oai': 'http://www.openarchives.org/OAI/2.0/',
    'arxiv': 'http://arxiv.org/OAI/arXiv/'
}
OAI_URL = "https://oaipmh.arxiv.org/oai"
START_DATE = '2021-01-01'
NUM_WORKERS = 2 #len(os.sched_getaffinity(0))
THROTTLE = 1
REQUEST_TIMEOUT = 30
LATEXPAND_TIMEOUT = 30
MAX_FAILURES = 3
CHUNK_SIZE = 200
TEX_KEYWORDS = [b"tikzpicture", b"circuitikz", b"tikzcd"]
SAVE_DIR = "ArXiv/oai"
EXTRACT_DIR = "ArXiv/arxiv_extracted"
ARXIV_ABS_URL = "https://arxiv.org/abs/"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)

record_count = 0


def is_critical_time():
    now = datetime.datetime.utcnow().time()
    return dtime(2, 0) <= now < dtime(4, 0)


extraction_buffer = []
def append_to_buffer(rec):
    global record_count
    extraction_buffer.append(rec)
    if len(extraction_buffer) >= CHUNK_SIZE:
        flush_extraction_buffer()


def flush_extraction_buffer():
    global record_count, extraction_buffer
    if not extraction_buffer:
        return
    idx = record_count // CHUNK_SIZE
    path = os.path.join(EXTRACT_DIR, f"arxiv_src_new_{idx}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        for rec in extraction_buffer:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    record_count += len(extraction_buffer)
    extraction_buffer.clear()


def sanitize_arxiv_id(aid):
    aid = re.sub(r'[\u2010-\u2015\u2212\uFE63\uFF0D]', '-', aid)
    return unicodedata.normalize("NFKD", aid).encode("ascii", "ignore").decode()


def filter_func(tex):
    return any(env in tex for env in TEX_KEYWORDS)


def find_tex_files(root_dir):
    return [str(p) for ext in ("*.tex", "*.pgf") for p in Path(root_dir).rglob(ext)]


def safe_extract_tarball(tarbytes, extract_path, aid):
    try:
        with tarfile.open(fileobj=io.BytesIO(tarbytes), mode="r:*") as tar:
            tar.extractall(path=extract_path)
        return True
    except tarfile.ReadError:
        return False
    except Exception:
        return False


def latexpand(tex_file_path):
    with NamedTemporaryFile(buffering=0) as tmp:
        path = os.path.dirname(tex_file_path) or "."
        file = os.path.basename(tex_file_path)
        cmd = ["latexpand", "--keep-comments", file, "--output", tmp.name]
        try:
            subprocess.run(cmd, cwd=path, stdout=DEVNULL, stderr=DEVNULL, check=True, timeout=LATEXPAND_TIMEOUT)
            tmp.seek(0)
            return tmp.read().decode("utf-8", errors="replace").strip()
        except Exception:
            return False


session = requests.Session()
# session.headers.update({"User-Agent": "tikz-harvester"})
session.headers.update({
    "User-Agent": "<YOUR_USER_AGENT>",
    "Accept": "<YOUR_ACCEPT_HEADER>"
})


def process_tarball(arxiv_id, tarbytes):
    try:
        failures = 0
        repo_fullname = sanitize_arxiv_id(arxiv_id)
        with tempfile.TemporaryDirectory() as tmpdir:
            if not safe_extract_tarball(tarbytes, tmpdir, arxiv_id):
                return
            tex_files = find_tex_files(tmpdir)
            for tex in tex_files:
                if failures > MAX_FAILURES:
                    return f"[!] Max failures reached in: {repo_fullname}"
                expanded = latexpand(tex)
                content_to_check = expanded
                if expanded is False:
                    failures += 1
                    try:
                        with open(tex, "r", encoding="utf-8", errors="replace") as f:
                            content_to_check = f.read()
                    except Exception:
                        continue
                if content_to_check and filter_func(content_to_check.encode("utf-8", errors="ignore")):
                    snippet = {
                        "text": content_to_check,
                        "meta": {
                            "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "yymm": repo_fullname[:4],
                            "arxiv_id": repo_fullname,
                            "url": f"{ARXIV_ABS_URL}{repo_fullname}",
                            "source": "arxiv"
                        }
                    }
                    append_to_buffer(snippet)
            if failures > MAX_FAILURES:
                return f"[!] Max failures reached in: {repo_fullname}"
            return f"[+] Processed: {repo_fullname}"
    except Exception:
        pass


def fetch_and_process(aid):
    cached_path = os.path.join(ROOT_DIR, "tarballs", f"{aid}.tar.gz")
    tarbytes = None
    if os.path.exists(cached_path):
        with open(cached_path, "rb") as f:
            tarbytes = f.read()
    else:
        url = f"https://arxiv.org/e-print/{aid}"
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                tarbytes = r.content
                os.makedirs(os.path.dirname(cached_path), exist_ok=True)
                with open(cached_path, "wb") as f:
                    f.write(tarbytes)
            else:
                return
        except Exception:
            return
    if tarbytes:
        process_tarball(aid, tarbytes)
        try:
            if os.path.exists(cached_path):
                os.remove(cached_path)
        except Exception:
            pass
        time.sleep(THROTTLE + random.uniform(0, 2))


def parse_record(record):
    try:
        arxiv_id = record.find('.//arxiv:id', namespaces=NAMESPACES).text
        created_str = record.find('.//arxiv:created', namespaces=NAMESPACES)
        if created_str is not None and created_str.text:
            created_date = datetime.datetime.strptime(created_str.text, "%Y-%m-%d").date()
        else:
            created_date = datetime.datetime.strptime(record.find('.//arxiv:created', namespaces=NAMESPACES).text, "%Y-%m-%d").date()
        return arxiv_id, created_date
    except Exception as e:
        return None, None


def month_range(start_date, end_date):
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    current = start
    while current < end:
        next_month = (current.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
        yield current.date(), (next_month - datetime.timedelta(days=1)).date()
        current = next_month


def fetch_records_monthly(from_date, until_date, resume_token_file):
    if is_critical_time():
        time.sleep(7200)
    token = None
    if os.path.exists(resume_token_file):
        with open(resume_token_file, "r") as f:
            token = f.read().strip()
    if token:
        params = {'verb': 'ListRecords', 'resumptionToken': token}
    else:
        params = {
            'verb': 'ListRecords',
            'metadataPrefix': 'arXiv',
            'from': from_date,
            'until': until_date,
        }
    while True:
        try:
            # resp = requests.get(OAI_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp = session.get(OAI_URL, params=params, timeout=REQUEST_TIMEOUT)
            time.sleep(THROTTLE + random.uniform(0, 2))
            resp.raise_for_status()
        except Exception:
            time.sleep(30)
            continue
        root = etree.fromstring(resp.content)
        records = root.findall('.//oai:record', namespaces=NAMESPACES)
        for rec in records:
            yield rec
        token_el = root.find('.//oai:resumptionToken', namespaces=NAMESPACES)
        if token_el is None or not token_el.text or token_el.text.strip() == '':
            if os.path.exists(resume_token_file):
                os.remove(resume_token_file)
            break
        token = token_el.text.strip()
        with open(resume_token_file, "w") as f:
            f.write(token)
        params = {'verb': 'ListRecords', 'resumptionToken': token}
        time.sleep(THROTTLE + random.uniform(0, 2))


if __name__ == "__main__":
    try:
        end_date = datetime.datetime.utcnow().date().isoformat()
        for month_from, month_until in month_range(START_DATE, end_date):
            token_file = os.path.join(SAVE_DIR, f"resume_{month_from}.token")
            futures = set()
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool, tqdm(desc=f"downloads ({month_from})", unit="src", dynamic_ncols=True, file=sys.stderr) as bar:
                for rec in fetch_records_monthly(month_from.isoformat(), month_until.isoformat(), token_file):
                    arxiv_id, created = parse_record(rec)
                    if arxiv_id is None:
                        continue
                    aid_clean = sanitize_arxiv_id(arxiv_id.split('v')[0])
                    if not (month_from <= created <= month_until):
                        continue
                    future = pool.submit(fetch_and_process, aid_clean)
                    futures.add(future)
                for future in as_completed(futures):
                    try:
                        _ = future.result()
                    except Exception:
                        continue
                    bar.update(1)
                time.sleep(300)
    except:
        flush_extraction_buffer()