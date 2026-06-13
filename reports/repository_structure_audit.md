# Recursive Repository Structure Audit

Found **1 findings**: 0 critical, 0 high, 0 medium, 1 low.

## Findings

### 1. [LOW] Local .env exists

- Area: `root`

- Detail: .env is present locally and ignored.

- Recommendation: OK locally; never commit secrets.

## Stats

```json
{
  "file_count_scanned": 433,
  "extension_counts": {
    ".csv": 171,
    ".ipynb": 104,
    ".py": 52,
    ".md": 45,
    ".json": 38,
    "<none>": 9,
    ".log": 9,
    ".png": 2,
    ".toml": 1,
    ".example": 1,
    ".yaml": 1
  },
  "top_level_sizes": {
    "_inbox": "1.9KB",
    "artifacts": "4.2GB",
    "checks": "114.0KB",
    "conf": "1.3KB",
    "data": "123.2MB",
    "docs": "181.6KB",
    "models": "0B",
    "notebooks": "4.3MB",
    "reports": "967.4KB",
    "scripts": "471.2KB",
    "src": "156.5KB",
    "submissions": "6.5MB",
    "tests": "12.2KB"
  },
  "submissions_files": 186,
  "submissions_size": "6.5MB",
  "submissions_categories": {
    "submission": 98,
    "embedding": 30,
    "oof": 55,
    "log": 1,
    "other": 1,
    "readme": 1
  }
}
```
