# Recursive Repository Structure Audit

Found **4 findings**: 0 critical, 0 high, 3 medium, 1 low.

## Findings

### 1. [MEDIUM] Generated/cache directories present

- Area: `generated`

- Detail: ['.pytest_cache', '.ruff_cache', 'tests/__pycache__', 'scripts/__pycache__', 'src/__pycache__', 'src/features/__pycache__', 'src/utils/__pycache__']

- Recommendation: Delete cache/generated directories before commit.

### 2. [MEDIUM] Unexpected root directories

- Area: `root`

- Detail: ['.obsidian', '.pytest_cache', '.ruff_cache']

- Recommendation: Move generated/tool-specific directories to ignored artifacts/ or delete them.

### 3. [MEDIUM] Unexpected root files

- Area: `root`

- Detail: ['.DS_Store']

- Recommendation: Move ad-hoc files into docs/, scripts/, reports/, or artifacts/.

### 4. [LOW] Local .env exists

- Area: `root`

- Detail: .env is present locally and ignored.

- Recommendation: OK locally; never commit secrets.

## Stats

```json
{
  "file_count_scanned": 3370,
  "extension_counts": {
    ".json": 1224,
    ".md": 542,
    ".py": 501,
    ".csv": 460,
    ".ipynb": 368,
    ".pyc": 221,
    "<none>": 28,
    ".log": 10,
    ".pdf": 5,
    ".tag": 2,
    ".png": 2,
    ".toml": 1,
    ".example": 1,
    ".before_20260621_daily_slate_20260621_162812": 1,
    ".yaml": 1,
    ".jsonl": 1,
    ".bib": 1,
    ".tex": 1
  },
  "top_level_sizes": {
    ".obsidian": "6.5KB",
    ".pytest_cache": "54.1KB",
    ".ruff_cache": "37.3KB",
    "_inbox": "1.9KB",
    "artifacts": "4.2GB",
    "build": "126.6MB",
    "checks": "114.0KB",
    "conf": "1.3KB",
    "data": "475.9MB",
    "docs": "1.1MB",
    "models": "0B",
    "notebooks": "4.8MB",
    "reports": "68.7MB",
    "scripts": "9.1MB",
    "src": "428.9KB",
    "submissions": "18.4MB",
    "tests": "2.1MB"
  },
  "submissions_files": 474,
  "submissions_size": "18.4MB",
  "submissions_categories": {
    "oof": 86,
    "submission": 355,
    "embedding": 30,
    "log": 1,
    "other": 1,
    "readme": 1
  }
}
```
