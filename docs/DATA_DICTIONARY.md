# 📊 데이터 명세서 (Data Dictionary)

## 1. 파일 목록 및 구조 (File Overview)
```text
data/
├── ch2025_data_items/
│ ├── ch2025_mACStatus.parquet
│ ├── ch2025_mActivity.parquet
│ ├── ch2025_mAmbience.parquet
│ ├── 총 12개의 스마트폰 및 웨어러블 센서(라이프로그) 데이터 (700일분)
│ └── ... (mBle, mGps, mLight, mScreenStatus, mUsageStats, mWifi, wHr, wLight, wPedo)
│
├── ch2026_metrics_description.pdf (지표 설명서)
├── ch2026_metrics_train.csv (Seven metrics, 450일분)
└── ch2026_submission_sample.csv
```

## 2. 세부 설명
본 경진대회에서는 라이프로그 데이터(총 12개 항목 700일분)와 레이블(7개 지표 450일분)을 제공합니다. 

- `ch2025_data_items/`: 라이프로그 데이터 (700일분). 스마트폰('m' 접두사) 및 웨어러블('w' 접두사)에서 수집된 다양한 센서 및 사용 통계 데이터(`parquet` 형식).
  - ETRI 데이터셋 링크: [ch2025_data_items (회원가입 필요)](https://epretx.etri.re.kr/dataDetail?lang=ko&id=459)
- `ch2026_metrics_train.csv`: 450일분의 학습용 레이블(정답) 데이터. 7가지 지표(설문지표 3개, 수면지표 4개)를 포함.
- `ch2026_metrics_description.pdf`: 7가지 지표(Seven metrics)에 대한 상세 설명 문서.
- `ch2026_submission_sample.csv`: 제출 양식 파일.
