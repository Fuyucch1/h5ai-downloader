#!/usr/bin/env python3
import asyncio
import argparse
import os
from pathlib import Path
from typing import Set, Tuple, List, Optional, Dict
from urllib.parse import urlparse, unquote

import httpx
from playwright.async_api import async_playwright, Page
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
)

# ===== h5ai selectors (skip parent "..") =====
H5AI_ITEM_SELECTOR = "ul#items li.item"
FOLDER_LINKS_SELECTOR = "ul#items li.item.folder:not(.folder-parent) > a"
FILE_LINKS_SELECTOR = "ul#items li.item.file a"

console = Console()

# ===== helpers =====
def safe_seg(s: str) -> str:
    s = unquote(s)
    return s.replace("/", "_").replace("\\", "_").strip()

async def wait_for_h5ai(page: Page, timeout_ms: int = 30000):
    await page.wait_for_selector(H5AI_ITEM_SELECTOR, timeout=timeout_ms)

async def extract_links(page: Page) -> Tuple[List[str], List[str]]:
    folders = await page.eval_on_selector_all(
        FOLDER_LINKS_SELECTOR,
        "els => els.map(e => new URL(e.getAttribute('href') || e.href, location.href).href)",
    )
    files = await page.eval_on_selector_all(
        FILE_LINKS_SELECTOR,
        "els => els.map(e => new URL(e.getAttribute('href') || e.href, location.href).href)",
    )
    return folders, files

def create_output_dir(start_url: str, base: Optional[Path]) -> Path:
    """
    downloads/<website>/<leaf-folder>
    <base>/<website>/<leaf-folder> if -o is given
    """
    p = urlparse(start_url)
    host = p.netloc
    parts = [seg for seg in p.path.split("/") if seg]
    leaf = parts[-1] if parts else host
    base_dir = base if base else Path("downloads")
    out = base_dir / safe_seg(host) / safe_seg(leaf)
    out.mkdir(parents=True, exist_ok=True)
    return out.resolve()

def rel_path_from_start(start_url: str, file_url: str) -> List[str]:
    s = urlparse(start_url); f = urlparse(file_url)
    if s.netloc != f.netloc:
        raise ValueError("Different host")
    s_parts = [seg for seg in s.path.split("/") if seg]
    f_parts = [seg for seg in f.path.split("/") if seg]
    if len(f_parts) < len(s_parts) + 1 or f_parts[: len(s_parts)] != s_parts:
        raise ValueError("Not under start folder")
    return [safe_seg(seg) for seg in f_parts[len(s_parts):]]

async def gather_all_files(page: Page, start_url: str, max_depth: int) -> List[str]:
    visited_dirs: Set[str] = set()
    files_to_get: List[str] = []

    async def dfs(url: str, depth: int):
        if url in visited_dirs or depth > max_depth:
            return
        visited_dirs.add(url)
        await page.goto(url)
        await wait_for_h5ai(page)
        folders, files = await extract_links(page)
        for f in files:
            try:
                _ = rel_path_from_start(start_url, f)
                files_to_get.append(f)
            except Exception:
                pass
        for sub in folders:
            try:
                _ = rel_path_from_start(start_url, sub)
            except Exception:
                continue
            await dfs(sub, depth + 1)

    await dfs(start_url, 0)
    return files_to_get

def build_cookie_header(playwright_cookies: List[dict], host: str) -> str:
    pairs = []
    for c in playwright_cookies:
        dom = c.get("domain") or host
        if host.endswith(dom.lstrip(".")):
            pairs.append(f"{c['name']}={c['value']}")
    return "; ".join(pairs)

# ===== probing & download =====
async def head_probe(client: httpx.AsyncClient, url: str) -> tuple[Optional[int], bool]:
    # HEAD, fallback to Range probe if needed
    try:
        r = await client.head(url)
        if r.status_code == 405:
            raise Exception("HEAD not allowed")
        r.raise_for_status()
        length = int(r.headers.get("Content-Length", "0")) or None
        can_range = r.headers.get("Accept-Ranges", "").lower().strip() == "bytes"
        return length, can_range
    except Exception:
        r = await client.get(url, headers={"Range": "bytes=0-0"})
        if r.status_code in (200, 206):
            length = None
            if "Content-Range" in r.headers:
                try:
                    length = int(r.headers["Content-Range"].split("/")[-1])
                except Exception:
                    pass
            return length, r.status_code == 206
        r.raise_for_status()
        return None, False

async def preprobe_sizes(client: httpx.AsyncClient, urls: List[str], concurrency: int) -> Dict[str, Optional[int]]:
    """
    Probe sizes concurrently so the OVERALL bar knows total bytes (and shows correct MB/s).
    """
    sem = asyncio.Semaphore(concurrency)
    sizes: Dict[str, Optional[int]] = {}

    async def one(u: str):
        async with sem:
            size, _ = await head_probe(client, u)
            sizes[u] = size

    await asyncio.gather(*[one(u) for u in urls])
    return sizes

async def ranged_download(client: httpx.AsyncClient, url: str, dest: Path,
                          total_size: int, segments: int,
                          progress: Progress, overall_task: int):
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_size = total_size // segments
    ranges = []
    start = 0
    for i in range(segments):
        end = (start + part_size - 1) if i < segments - 1 else (total_size - 1)
        ranges.append((start, end))
        start = end + 1

    file_task = progress.add_task(f"[cyan]{dest.name}[/]", total=total_size)
    part_paths = [dest.with_suffix(dest.suffix + f".part.{i}") for i in range(segments)]
    tmp_path = dest.with_suffix(dest.suffix + ".part")

    async def dl(i: int, byte_range: tuple[int, int]):
        rs, re = byte_range
        headers = {"Range": f"bytes={rs}-{re}"}
        async with client.stream("GET", url, headers=headers) as r:
            r.raise_for_status()
            with open(part_paths[i], "wb") as f:
                async for chunk in r.aiter_bytes(1024 * 1024):
                    if not chunk: continue
                    f.write(chunk)
                    n = len(chunk)
                    progress.update(file_task, advance=n)
                    progress.update(overall_task, advance=n)

    await asyncio.gather(*[dl(i, ranges[i]) for i in range(segments)])

    # merge parts
    with open(tmp_path, "wb") as out:
        for p in part_paths:
            with open(p, "rb") as src:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk: break
                    out.write(chunk)
    for p in part_paths:
        try: p.unlink()
        except Exception: pass
    os.replace(tmp_path, dest)
    progress.remove_task(file_task)

async def single_stream_download(client: httpx.AsyncClient, url: str, dest: Path,
                                 progress: Progress, overall_task: int):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    async with client.stream("GET", url) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0")) or None
        file_task = progress.add_task(f"[cyan]{dest.name}[/]", total=total)
        with open(tmp, "wb") as f:
            async for chunk in r.aiter_bytes(1024 * 1024):
                if not chunk: continue
                f.write(chunk)
                n = len(chunk)
                progress.update(file_task, advance=n)
                progress.update(overall_task, advance=n)
    os.replace(tmp, dest)
    progress.remove_task(file_task)

async def worker(queue: asyncio.Queue, client: httpx.AsyncClient,
                 progress: Progress, overall_task: int,
                 start_url: str, out_dir: Path, segments: int):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        file_url = item
        try:
            rel = rel_path_from_start(start_url, file_url)
            dest = out_dir.joinpath(*rel)
            length, can_range = await head_probe(client, file_url)
            if can_range and length and length > 4 * 1024 * 1024 and segments > 1:
                await ranged_download(client, file_url, dest, length, segments, progress, overall_task)
            else:
                await single_stream_download(client, file_url, dest, progress, overall_task)
        except Exception as e:
            progress.console.print(f"[red]✗ {file_url}[/] — {e}")
        finally:
            queue.task_done()

# ===== main =====
async def main():
    ap = argparse.ArgumentParser(description="H5ai folder downloader")
    ap.add_argument("url", help="Root h5ai folder URL")
    ap.add_argument("-o", "--out", default=None,
                    help="Base dir (default: ./downloads). Final path: downloads/<website>/<folder>")
    ap.add_argument("--concurrency", type=int, default=8, help="Parallel files (default: 8)")
    ap.add_argument("--segments", type=int, default=8, help="Parallel segments per large file (default: 8)")
    ap.add_argument("--max-depth", type=int, default=999999, help="Max recursion depth (default: unlimited)")
    ap.add_argument("--timeout", type=int, default=30, help="Playwright page timeout (default: 30s)")
    ap.add_argument("--connect-timeout", type=float, default=30.0, help="HTTP connect timeout seconds (default: 30s)")
    ap.add_argument("--read-timeout", type=float, default=0.0,
                    help="HTTP read timeout seconds (default: 0 (no read timeout))")
    ap.add_argument("--headful", action="store_true", help="Run browser non-headless")
    args = ap.parse_args()

    base_dir = Path(args.out).resolve() if args.out else None

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not args.headful)
        context = await browser.new_context(accept_downloads=False)
        page = await context.new_page()
        page.set_default_timeout(args.timeout * 1000)
        context.set_default_timeout(args.timeout * 1000)

        try:
            console.print("Parsing the url and gathering file list. Please wait...")
            files = await gather_all_files(page, args.url, args.max_depth)
            console.print(f"Found {len(files)} items to download!")

            out_dir = create_output_dir(args.url, base_dir)

            ua = await page.evaluate("navigator.userAgent")
            host = urlparse(args.url).netloc
            cookie_header = build_cookie_header(await context.cookies(), host)
            headers = {"User-Agent": ua, "Referer": args.url}
            if cookie_header:
                headers["Cookie"] = cookie_header

            read_timeout = None if args.read_timeout == 0 else args.read_time_out if hasattr(args, "read_time_out") else args.read_timeout
            timeout = httpx.Timeout(
                connect=args.connect_timeout,
                read=read_timeout,
                write=None,
                pool=None,
            )
            limits = httpx.Limits(
                max_keepalive_connections=max(args.concurrency * 4, 16),
                max_connections=max(args.concurrency * 8, 32),
            )

            async with httpx.AsyncClient(
                headers=headers, timeout=timeout, limits=limits, follow_redirects=True, http2=True
            ) as client:

                size_map = await preprobe_sizes(client, files, concurrency=max(8, args.concurrency * 2))
                total_known = sum(s for s in size_map.values() if s)

                queue: asyncio.Queue = asyncio.Queue()
                for f in files:
                    queue.put_nowait(f)

                # Progress UI
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                )
                with progress:
                    overall = progress.add_task("[magenta]Overall[/]",
                                                total=(total_known if total_known > 0 else None))

                    workers = [
                        asyncio.create_task(
                            worker(queue, client, progress, overall, args.url, out_dir, args.segments)
                        )
                        for _ in range(args.concurrency)
                    ]
                    await queue.join()
                    for _ in workers:
                        queue.put_nowait(None)
                    await asyncio.gather(*workers)

        finally:
            try:
                for p in context.pages:
                    await p.close()
            except Exception:
                pass
            await context.close()
            await browser.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\nInterrupted.")
