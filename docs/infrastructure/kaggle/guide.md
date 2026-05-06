# Kaggle CLI (`kaggle`) 활용 가이드

Kaggle CLI는 캐글 대회뿐만 아니라 DACON 대회에서도 **무료 클라우드 GPU 자원(Kaggle Notebooks)**을 활용하기 위해 데이터와 코드를 동기화하는 자동화(Automation) 파이프라인의 핵심 도구입니다.

---

## 1. 인증 정보 구성
본 베이스라인에는 계정 인증 정보가 안전하게 세팅되어야 합니다.
- Kaggle 웹사이트 -> Settings -> Create New Token에서 다운로드한 `kaggle.json` 파일을 사용자 홈 디렉토리(`~/.kaggle/kaggle.json`)에 두고 `chmod 600` 권한을 부여합니다.
- 프로젝트 디렉토리 최상단에 두더라도 `.gitignore` 규칙에 의해 절대로 Git 저장소에 커밋되지 않습니다.

---

## 2. 주요 CLI 명령어 (실무 파이프라인)

클라우드 훈련을 위해 로컬 코드를 Kaggle 서버로 밀어넣는(Push) 3단계 워크플로우입니다.

### Step 1. 로컬 코드를 Kaggle Dataset으로 만들기
Kaggle 커널이 내 코드를 `import` 해서 쓸 수 있도록, 소스 폴더(`src/`, `scripts/`) 자체를 데이터셋으로 업로드합니다.
```bash
# 최초 1회 초기화 및 업로드
kaggle datasets init -p ./src
kaggle datasets create -p ./src

# 코드 수정 후 업데이트 할 때 (가장 자주 쓰는 명령어)
kaggle datasets version -p ./src -m "Add new Time-Series features"
```

### Step 2. 백그라운드 훈련 커널(노트북) 실행
로컬의 훈련 설정 파일(`kernel-metadata.json`)을 바탕으로 클라우드에서 노트북을 실행합니다. 터미널을 꺼도 캐글 서버에서 계속 돌아갑니다.
```bash
# 백그라운드 훈련 시작
kaggle kernels push -p ./kaggle_notebook_folder
```

### Step 3. 학습 완료 결과물 내려받기
웹 브라우저를 켜지 않고도 CLI로 훈련된 가중치 파일(`.pt`)과 `submission.csv` 파일을 다운로드합니다.
```bash
# 특정 커널의 결과물 전체 다운로드
kaggle kernels output softkleenex/my-dacon-training-kernel -p ./models
```

---

## 3. 꿀팁 (Tips)
- **외부 데이터 확보**: DACON 대회에서 제공하는 데이터셋 외에 추가 데이터가 필요할 때, 직접 크롤링할 필요 없이 누군가 캐글에 올려둔 데이터를 `kaggle datasets download -d [유저명/데이터셋명]`으로 1초 만에 가져와 시간을 절약하세요.
- **상태 확인**: `kaggle kernels status softkleenex/my-dacon-training-kernel` 명령어로 현재 내 훈련 코드가 클라우드에서 실행 중인지, 실패(Error)했는지 터미널에서 바로 확인할 수 있습니다.
