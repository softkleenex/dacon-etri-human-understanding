# 📊 데이터 명세서 (Data Dictionary)

## 1. 파일 목록 및 구조 (File Overview)
```text
data/
└── raw/
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

> ⚠️ **주의 (Data & Metrics Year Anomaly)**: 
> 제공된 데이터 폴더명(`ch2025_data_items`) 및 공식 문서는 2025년을 가리키지만, 제공된 레이블 파일명(`ch2026_metrics_train.csv`) 및 지표 문서는 2026년을 가리킵니다. 이는 ETRI 대회 주최측의 연도 표기 불일치로 보이나, 동일한 데이터셋 세트의 입력(2025 라이프로그)과 타겟(2026 레이블)으로 간주하고 진행합니다.

- `ch2025_data_items/`: 라이프로그 데이터 (700일분). 스마트폰('m' 접두사) 및 웨어러블('w' 접두사)에서 수집된 다양한 센서 및 사용 통계 데이터(`parquet` 형식).
  - ETRI 데이터셋 링크: [ch2025_data_items (회원가입 필요)](https://epretx.etri.re.kr/dataDetail?lang=ko&id=459)
  
### 📱 센서 데이터 명세 (12 Sensor Parquet Files)

각 파일은 참가자 ID와 타임스탬프(KST 기준, YYYY-MM-DD HH:MM:SS)를 포함합니다. 센서 측정은 스마트폰이나 스마트워치 미착용, 배터리 방전 등으로 인한 결측치(Missing Values)와 노이즈가 다수 존재할 수 있습니다.

| 파일명 (File Name) | 센서 기기 | 주요 컬럼 (Key Columns) | 데이터 타입 (Data Type) | 설명 (Description & Values) |
|---|---|---|---|---|
| `ch2025_mACStatus.parquet` | 스마트폰 | `m_charging` | integer | 충전 상태 (0: No, 1: Charging) |
| `ch2025_mActivity.parquet` | 스마트폰 | `m_activity` | integer | 사용자 활동 상태 (0: in_vehicle, 1: on_bicycle, 2: on_foot, 3: still, 4: unknown, 5: tilting, 7: walking, 8: running) |
| `ch2025_mAmbience.parquet` | 스마트폰 | `m_ambience` | object | 주변 소음 레이블과 해당 확률 리스트 (List) |
| `ch2025_mBle.parquet` | 스마트폰 | `m_ble` | object | 주변 블루투스 기기 주소, device_class, rssi 리스트 (List) |
| `ch2025_mGps.parquet` | 스마트폰 | `m_gps` | object | 고도, 위도, 경도, 속도 리스트 (상대 좌표로 익명화됨) (List) |
| `ch2025_mLight.parquet` | 스마트폰 | `m_light` | float | 주변 조도 (단위: lx) |
| `ch2025_mScreenStatus.parquet`| 스마트폰 | `m_screen_use` | integer | 화면 사용 여부 (0: No, 1: Using screen) |
| `ch2025_mUsageStats.parquet` | 스마트폰 | `m_usage_stats` | object | 사용한 앱 이름 및 사용 시간(ms) 리스트 (List) |
| `ch2025_mWifi.parquet` | 스마트폰 | `m_wifi` | object | 주변 Wi-Fi bssid 및 rssi 리스트 (List) |
| `ch2025_wHr.parquet` | 스마트워치| `heart_rate` | object | 심박수 기록 리스트 (List) |
| `ch2025_wLight.parquet` | 스마트워치| `w_light` | float | 주변 조도 (단위: lx) |
| `ch2025_wPedo.parquet` | 스마트워치| `burned_calories`<br>`distance`<br>`speed`<br>`step`<br>`step_frequency` | float<br>float<br>float<br>integer<br>float | 소모 칼로리<br>이동 거리(m)<br>속도(km/h)<br>걸음 수<br>분당 걸음 빈도 |

- `ch2026_metrics_train.csv`: 450일분의 학습용 레이블(정답) 데이터. 7가지 지표(설문지표 3개, 수면지표 4개)를 포함.
- `ch2026_metrics_description.pdf`: 7가지 지표(Seven metrics)에 대한 상세 설명 문서.
- `ch2026_submission_sample.csv`: 제출 양식 파일.
