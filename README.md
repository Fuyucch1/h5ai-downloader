# h5ai-downloader

High-speed recursive downloader for [h5ai](https://larsjung.de/h5ai/) directory listings. It parses all files under a link recursively, and downloads them.

---

## Features

- Recursively crawls subfolders 
- **Parallel downloads** across multiple files

---

## Requirements

```bash
git clone https://github.com/Fuyucch1/h5ai-downloader
cd h5ai-downloader
pip install -r requirements.txt
playwright install
```

---

## Usage

```bash

python h5ai_downloader.py <URL> [options]
```

Example:


---

## ⚙️ Options

| Option | Description | Default |
|--------|--------------|----------|
| `-o`, `--out` | Base output directory | `./downloads` |
| `--concurrency` | Number of files downloaded simultaneously | `8` |
| `--segments` | Parallel segments per large file | `8` |
| `--max-depth` | Maximum recursion depth | unlimited |
| `--timeout` | Playwright page timeout (seconds) | `30` |
| `--connect-timeout` | Connection timeout for HTTP | `30` |
| `--read-timeout` | Read timeout (0 = disabled, safe for huge files) | `0` |
| `--headful` | Show browser window | off |

---

## 📦 Output Structure

```
downloads/
 └── <website>/
      └── <h5ai-folder>/
           ├── subfolder1/
           ├── subfolder2/
           └── files...
```

---

## Notes

- If you encounter slowdowns, try lowering `--segments` to 4 or `--concurrency` to 4–6.

---

## 📜 License

MIT License — free to use, modify, and share.
