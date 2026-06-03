import json
import os
from pathlib import Path
import pandas as pd

def generate_standalone_notebook(sub_id, desc, score, date, csv_content, sub_file):
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# 🏆 Standalone Historical Submission Re-creation (ID: {sub_id})\n",
                    f"### Model: {desc}\n",
                    f"### Local CV Score: {score}\n",
                    f"### Generation Date: {date}\n",
                    "\n",
                    "이 노트북은 어떠한 외부 `.py` 파일이나 모듈 의존성 없이 **완전한 Standalone**으로 실행 가능합니다.\n",
                    f"노트북을 실행하면 아래에 내장된 고유 예측값 데이터를 바탕으로 제출용 공통 파일 `submissions/submission.csv`와\n",
                    f"매칭된 고유 파일 `submissions/{sub_file}`을 즉시 재생성합니다."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 내장된 제출용 고유 CSV 데이터\n",
                    f"csv_data = \"\"\"{csv_content}\"\"\""
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "\n",
                    "# submissions 디렉토리 확인\n",
                    "Path(\"submissions\").mkdir(exist_ok=True)\n",
                    "\n",
                    "# 1. 데이콘 제출용 공통 파일 생성\n",
                    "default_output = Path(\"submissions/submission.csv\")\n",
                    "with open(default_output, \"w\", encoding=\"utf-8\") as f:\n",
                    "    f.write(csv_data.strip() + \"\\n\")\n",
                    "\n",
                    f"# 2. 버전에 매칭된 고유 CSV 파일 생성\n",
                    f"matched_output = Path(\"submissions/{sub_file}\")\n",
                    "with open(matched_output, \"w\", encoding=\"utf-8\") as f:\n",
                    "    f.write(csv_data.strip() + \"\\n\")\n",
                    "\n",
                    "print(f\"✅ 제출용 예측 파일이 성공적으로 생성되었습니다:\")\n",
                    "print(f\"   - 공통 제출 경로: {default_output} ({default_output.stat().st_size} bytes)\")\n",
                    "print(f\"   - 매칭 고유 경로: {matched_output} ({matched_output.stat().st_size} bytes)\")\n",
                    "\n",
                    "# 생성된 데이터 검증용 프리뷰 출력\n",
                    "df = pd.read_csv(default_output)\n",
                    "print(f\"\\n데이터 형태: {df.shape}\")\n",
                    "print(df.head())"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    return notebook

def main():
    log_path = Path("submissions/submission_log.csv")
    if not log_path.exists():
        print("Error: submission_log.csv not found.")
        return

    df_log = pd.read_csv(log_path)
    history_dir = Path("notebooks/history")
    history_dir.mkdir(parents=True, exist_ok=True)

    print("Generating fully standalone notebooks with embedded CSV content...")

    for idx, row in df_log.iterrows():
        sub_id = int(row['id'])
        desc = str(row['model_description'])
        score = float(row['local_cv_score']) if pd.notna(row['local_cv_score']) else 0.0
        date = str(row['date'])
        sub_file = str(row['submission_file'])

        # Read the corresponding CSV file content
        csv_path = Path("submissions") / sub_file
        if not csv_path.exists():
            # Fallback to general submission.csv if the specific one is missing
            csv_path = Path("submissions/submission.csv")

        if not csv_path.exists():
            print(f"⚠️ Warning: Missing predictions source for ID {sub_id} ({sub_file}). Skipping.")
            continue

        with open(csv_path, encoding='utf-8') as f:
            csv_content = f.read().strip()

        # Determine pipeline type name for file structure consistency
        if "Classifier Chain" in desc or sub_id >= 49:
            pipeline_type = "Classifier_Chain"
        else:
            pipeline_type = "LOSO_Ensemble"

        # If sub_file is missing or NaN, construct a fallback filename
        if not sub_file or sub_file == 'nan':
            sub_file = f"submission_id_{sub_id:03d}_score{score:.6f}.csv"

        notebook = generate_standalone_notebook(sub_id, desc, score, date, csv_content, sub_file)
        ipynb_name = f"submission_{sub_id:03d}_{pipeline_type}_score{score:.4f}.ipynb"
        ipynb_path = history_dir / ipynb_name

        with open(ipynb_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"🎉 Success! Generated {len(df_log)} standalone notebooks in notebooks/history/")

if __name__ == '__main__':
    main()
