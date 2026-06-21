import asyncio
import httpx
import json
import time
import random


SEEN_PATH = "GitHub/seen_repos.json"
EXCLUDED_PATH = "GitHub/excluded_repos.json"
KEYS_PATH = "GitHub/github_key.json"
SHARDS = ['"tikz" in:file extension:tex']


def load_seen_repos(filepath=SEEN_PATH):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except:
        return set()


def load_excluded_repos(filepath=EXCLUDED_PATH):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except:
        return set()


def save_seen_repos(seen, filepath=SEEN_PATH):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(list(seen), f, indent=2, ensure_ascii=False)


async def fetch_with_rate_limit(client, url, token_index, tokens):
    while True:
        resp = await client.get(url, timeout=60)
        # remaining = int(resp.headers.get("X-RateLimit-Remaining", 0))
        reset_time = int(resp.headers.get("X-RateLimit-Reset", 0))
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            wait_seconds = max(reset_time - int(time.time()), 90)
            await asyncio.sleep(wait_seconds)
            token_index = (token_index + 1) % len(tokens)
            client.headers.update({"Authorization": f"Bearer {tokens[token_index]}"})
            continue
        return resp, token_index


async def gather_forever(tokens, shard_queries):
    seen_repos = load_seen_repos()
    excluded_repos = load_excluded_repos()
    token_index = 0
    headers = {"Authorization": f"Bearer {tokens[token_index]}"}
    async with httpx.AsyncClient(headers=headers) as client:
        while True:
            for shard in shard_queries:
                page = 1
                while True:
                    query = f"{shard}&per_page=100&page={page}&sort=indexed"
                    url = f"https://api.github.com/search/code?q={query}"
                    resp, token_index = await fetch_with_rate_limit(client, url, token_index, tokens)
                    if resp.status_code != 200:
                        break
                    results = resp.json().get("items", [])
                    if not results:
                        break
                    found_new = False
                    for item in results:
                        repo = item["repository"]["full_name"]
                        if repo in seen_repos or repo in excluded_repos:
                            continue
                        seen_repos.add(repo)
                        found_new = True
                        break
                    if not found_new:
                        break
                    page += 1
                    await asyncio.sleep(random.uniform(1, 3))
                save_seen_repos(seen_repos)
            await asyncio.sleep(10)


if __name__ == "__main__":
    with open(KEYS_PATH, "r") as file:
        github_key = json.load(file)
    tokens = [github_key["keys"][f"github_key_{i + 1}"] for i in range(len(github_key["keys"]))]
    asyncio.run(gather_forever(tokens, SHARDS))