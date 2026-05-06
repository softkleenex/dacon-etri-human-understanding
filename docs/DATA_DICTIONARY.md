# 📊 데이터 명세서 (Data Dictionary)

## 1. 파일 목록 및 구조 (File Overview)
```text
├── train/
│  ├── TRAIN_00001.csv
│  ├── TRAIN_00002.csv
│  ├── ...
│  └── TRAIN_10000.csv
├── test/
│  ├── TEST_00001.csv
│  ├── TEST_00002.csv
│  ├── ...
│  └── TEST_10000.csv
├── train_labels.csv
└── sample_submission.csv
```

## 2. 세부 설명
본 대회의 좌표는 방 기준 절대 좌표가 아니라, LiDAR 센서 기준의 sensor-local 3차원 좌표계입니다. 좌표 단위는 미터(m)이며, 축 방향은 x: forward, y: left, z: up으로 정의됩니다.
- `train/`: 학습용 관측 시계열 샘플 폴더. 각 CSV는 하나의 모기 샘플이며, 컬럼은 `timestep_ms`, `x`, `y`, `z` 입니다.
- `test/`: 평가용 관측 시계열 샘플 폴더. 구조와 컬럼은 `train/`과 동일합니다.
- `train_labels.csv`: 각 `TRAIN_xxxxx.csv`에 대응되는 정답 파일. 컬럼은 `id`, `x`, `y`, `z` 이며, 마지막 관측 시점 이후 80ms 시점의 실제 모기 좌표를 의미합니다.
- `sample_submission.csv`: 제출 양식 파일. 컬럼은 `id`, `x`, `y`, `z` 입니다.