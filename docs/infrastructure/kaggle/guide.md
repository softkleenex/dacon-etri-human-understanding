# Kaggle CLI (`kaggle`) 활용 가이드

캐글 CLI는 DACON 대회뿐만 아니라 클라우드 리소스 활용을 위해 데이터를 다운로드하고 워크플로우를 자동화하는 필수 도구입니다.

## 1. 인증 정보 구성
- Kaggle 웹사이트 -> Settings -> Create New Token 메뉴에서 `kaggle.json` 다운로드.
- 해당 파일을 사용자 홈 디렉토리 하위의 `.kaggle/` 폴더에 복사합니다.
  ```bash
  mkdir -p ~/.kaggle
  cp ~/Downloads/kaggle.json ~/.kaggle/
  chmod 600 ~/.kaggle/kaggle.json
  ```
- 이 템플릿 환경에서는 프로젝트 폴더 최상단의 `kaggle.json` 파일은 `.gitignore` 처리되어 절대로 Git에 올라가지 않습니다.

## 2. 주요 CLI 명령어 (Automation)

**데이터셋 업로드 (로컬 코드를 클라우드로 보내기)**
```bash
# 최초 생성 시
kaggle datasets init -p ./src
# dataset-metadata.json 수정 후
kaggle datasets create -p ./src

# 코드 업데이트 시
kaggle datasets version -p ./src -m "Update code features"
```

**커널(노트북) 실행 (백그라운드 학습 트리거)**
```bash
# 노트북 메타데이터와 코드 푸시 및 실행
kaggle kernels push -p ./kaggle_notebook_folder
```

**학습 결과물 다운로드 (모델 가중치 & CSV 가져오기)**
```bash
# 실행 완료된 커널의 결과물만 내려받기
kaggle kernels output softkleenex/my-dacon-training-kernel -p ./models
```

## 3. 팁
대회 데이터 용량이 너무 크면 데이콘 사이트에서 다운받기보다, 누군가 Kaggle Dataset으로 미리 올려둔 것을 `kaggle datasets download -d [이름]` 명령어로 빠르게 다운받아 활용하는 것도 네트워크 리소스를 아끼는 좋은 꿀팁입니다.
