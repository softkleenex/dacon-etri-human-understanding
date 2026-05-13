# DACON 제출 API (`dacon_submit_api`) 활용 매뉴얼

DACON 웹페이지에 매번 접속해서 CSV 파일을 마우스로 업로드하는 번거로움을 없애고, 훈련 파이프라인의 맨 마지막 단계에서 결과물을 터미널/스크립트로 자동 제출하기 위해 사용합니다.

## 🎯 사용 목적 (Why use this?)
- **제출 파이프라인 자동화**: 로컬이나 클라우드에서 모델 추론(Inference)이 끝나면, 곧바로 DACON 리더보드에 채점을 요청하도록 파이프라인을 끊김 없이(End-to-End) 연결합니다.
- **Agent 연동 (자율 루프)**: Gemini CLI가 모델을 수정한 뒤 스스로 제출까지 완료하고, 웹 리더보드 점수를 기다리는 휴먼-인-더-루프(HITL) 워크플로우의 핵심 연결 고리입니다.

## 🚀 사용 방법 (How to use?)

### 1. 인증 및 환경 변수 설정
**이 템플릿 환경에서는 시스템 전역(`~/.zshrc`)에 `DACON_TOKEN`과 `DACON_TEAMNAME`이 미리 저장(Set Globally)되어 있습니다.**
따라서 파이썬 스크립트 실행 시 매번 토큰이나 팀명을 소스 코드에 절대 하드코딩할 필요가 없습니다.

```bash
# 전역 설정이 정상적으로 로드되었는지 확인하는 명령어
echo $DACON_TOKEN
echo $DACON_TEAMNAME
```

> **사전 패키지 설치 필요**: DACON에서 제공하는 `dacon_submit_api` 휠(whl) 파일이 설치되어 있어야 합니다. 새로운 가상 환경을 구성할 때마다 터미널에 아래 명령어를 입력하여 로컬에 영구 백업된 파일로 손쉽게 설치하세요.
> ```bash
> pip install ~/sysp/dacon_api/dacon_submit_api-0.0.4-py3-none-any.whl
> ```

### 2. 실전 파이썬 자동 제출 스니펫 및 Makefile 활용
대회의 성격(일반 CSV 제출 vs 코드 파일 제출)에 따라 사용하는 함수가 다릅니다. 이 템플릿에서는 **`make dacon-submit`** 명령어를 제공하여, 스크립트 작성 없이도 터미널에서 즉각적인 파일 제출이 가능합니다.

```bash
# 터미널에서 아래 명령을 실행하면 경로와 메모를 묻는 프롬프트가 나타납니다.
make dacon-submit
# 프롬프트 예시:
# Enter submission file path: submissions/xgb_fold5_ensemble.csv
# Enter submission message: Auto-submitted via baseline
```

추론 스크립트의 맨 마지막에 파이썬 로직을 덧붙여 파이프라인을 완전 자동화하려면 아래 코드를 참고하세요.

```python
import os
from dacon_submit_api import dacon_submit_api 

def submit_to_dacon(filepath: str, competition_id: str, submission_type: str = "general", memo: str = "Auto-submitted via baseline"):
    """
    전역 환경변수를 읽어와 DACON에 자동으로 제출합니다.
    
    :param filepath: 제출할 파일의 경로 (CSV 파일 또는 코드 파일/압축본)
    :param competition_id: 대회 URL에서 확인 가능한 대회 ID
    :param submission_type: "general" (일반 제출, 최대 100MB) 또는 "code" (코드 제출)
    """
    
    # 1. 전역 시스템 변수에서 토큰과 팀명 로드
    dacon_token = os.environ.get("DACON_TOKEN")
    team_name = os.environ.get("DACON_TEAMNAME")
    
    if not dacon_token or not team_name:
        raise ValueError("전역 환경변수(DACON_TOKEN, DACON_TEAMNAME)가 설정되지 않았습니다.")
        
    print(f"🚀 {team_name} 팀의 결과물을 제출합니다... (방식: {submission_type})")
    
    # 2. DACON API 호출 (제출 방식에 따른 분기)
    if submission_type == "general":
        result = dacon_submit_api.post_submission_file(
            file_path=filepath, 
            token=dacon_token, 
            cpt_id=competition_id, 
            team_name=team_name, 
            memo=memo
        )
    elif submission_type == "code":
        result = dacon_submit_api.post_code_submission_file(
            file_path=filepath, 
            token=dacon_token, 
            cpt_id=competition_id, 
            team_name=team_name, 
            memo=memo
        )
    else:
        raise ValueError("submission_type은 'general' 또는 'code'여야 합니다.")
    
    # 3. 결과 확인
    if result.get('isSubmitted'):
        print("✅ 제출 성공! 데이콘 웹페이지에서 리더보드를 확인하세요.")
    else:
        print(f"❌ 제출 실패: {result.get('detail')}")

# 사용 예시 (일반 결과물 제출)
# submit_to_dacon("submissions/xgb_fold5_ensemble.csv", "236690", submission_type="general")

# 사용 예시 (코드 제출 대회용)
# submit_to_dacon("submissions/my_code_submission.zip", "236690", submission_type="code")
```

## ⚠️ 실전 주의 사항
- **제출 파일 크기 제한**: 
  - 일반 제출(`general`): **100MB 이하**의 CSV 파일만 가능합니다.
  - 코드 제출(`code`): 각 대회별 상세페이지의 규정을 반드시 참고하세요.
- **하루 제출 횟수 제한**: DACON은 보통 하루 제출 횟수(예: 3회)가 엄격하게 정해져 있습니다. 무턱대고 API 제출을 자동화해두면 아까운 기회를 날릴 수 있습니다. **반드시 로컬 K-Fold OOF 점수가 기존 최고 점수를 갱신했을 때만 API가 호출되도록 제어문(if 조건)을 걸어두세요.**
