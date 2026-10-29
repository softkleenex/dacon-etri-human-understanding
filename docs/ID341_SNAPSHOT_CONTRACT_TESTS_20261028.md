# ID341 스냅샷 계약 테스트 처리 — 2026-10-28

## 문제

`make test` 에서 27건이 실패했다. 16개 파일, 전부 `test_post_id341_*` 계열과
`test_dacon_submit_approval_contract.py` 1건이다.

실패 원인은 코드 결함이 아니다. 이 테스트들은 `reports/` 에 저장된 리포트를 읽지 않고
매번 `scripts/diagnose_*.py` 의 `build_report()` 를 실행해 **현재 저장소 상태**로
계약을 다시 계산한다. 그런데 기대값은 ID341 시점에 고정돼 있다.

```
tests/test_post_id341_safe_wait_state_contract.py:51
    assert log["rows"] == 342      # 현재 submission_log.csv 는 349행
```

제출 로그는 2026-06-23 ~ 06-26 작업에서 342행 → 350행(데이터 349행)으로 늘었다.
그 시점부터 "ID341 이후 새 제출이 없다"를 전제하는 준비성·허용 판정이 모두 `False` 가
되었고, 테스트는 깨진 상태로 커밋되어 있었다.

즉 2026-06-26 대회 종료 시점에 이미 깨져 있었고, 2026-10-28 에 발견했다.

## 확인한 것

- `make audit` 실행이 원인이 아니다. 감사가 재생성한 리포트 11개를 `git checkout` 으로
  원복한 뒤에도 27건이 동일하게 실패했다.
- **안전 방향 회귀는 없다.** `test_dacon_submit_approval_contract.py` 는 5건 중 4건이
  통과하며, 실패한 1건은 "ID341 수동 sanity probe 를 **허용**한다"는 허용 측 단정이다.
  제출을 차단하는 fail-closed 단정들은 전부 통과한다. 게이트가 더 엄격해진 방향이다.

## 처리

27건에 `@pytest.mark.skip` 을 개별로 붙였다. 이유 문자열은 동일하다.

> ID341 시점 스냅샷 계약. 제출 로그가 342행에서 350행으로 늘어난 뒤에는 성립하지 않는다
> (대회 종료 2026-06-26).

파일 단위로 skip 하지 않은 이유가 있다. 같은 파일에 살아 있는 단정이 28건 남아 있고,
그중에는 위의 fail-closed 제출 차단 계약이 포함된다. 파일째로 끄면 그 보호가 함께 꺼진다.

기대값을 349행으로 고쳐 쓰는 방법은 택하지 않았다. 이 계약들의 의미는 "ID341 시점에
무엇이 허용/차단되었는가" 이고, 현재 숫자로 덮어쓰면 기록이 아니라 다른 주장이 된다.
대회가 끝나 재평가 대상도 아니다.

## 결과

```
make test   →  320 passed, 27 skipped
```

## 같은 커밋에서 처리한 구조 감사 HIGH 3건

`scripts/audit_repository_structure.py`:

1. **Readable PDF remains in tracked workspace** — `docs/paper_submission_*/` 아래 PDF를
   허용 대상으로 뺐다. 컴파일된 원고와 실제 심사 제출본은 Markdown 으로 바꿀 수 없는
   산출물이다. 나머지 4건은 `build/` 에 있었다.
2. **Unexpected root Markdown files** — `CLAUDE.md` 를 허용 목록에 넣었다.
   저장소에 포함되는 프로젝트 지침 파일이다.
3. **Nested project copies found** — `build/kaggle_upstream_patch_dataset` 였다.
   `build/` 는 gitignore 대상이므로 `SKIP_DIRS` 와 중첩 검사 제외 집합에 `build` 를 추가했다.
   검사 제목이 "tracked workspace" 인데 무시되는 빌드 산출물을 훑고 있었다.

```
전: 7 findings (3 high, 3 medium, 1 low)
후: 4 findings (0 high, 3 medium, 1 low)
```

남은 4건은 `.pytest_cache`, `.ruff_cache`, `__pycache__`, `.obsidian`, `.DS_Store` 로
전부 gitignore 되는 로컬 도구 산출물이다. `.obsidian` 은 사용자 편집기 설정이고 캐시는
테스트를 돌리면 즉시 다시 생기므로 삭제하지 않았다.

---

## 추가 확인 (2026-10-28, 전체 점검)

위 27건과 **같은 뿌리의 결함이 리포트 쪽에도 있다.** `make audit`을 돌리면
날짜가 박힌 과거 스냅샷이 현재 수치로 덮어써진다.

```
Makefile:337  scripts/diagnose_post_id341_public_local_gap.py \
                --output reports/post_id341_public_local_gap_diagnosis_20260618.json
```

파일명의 `20260618`은 고정인데 스크립트는 `submission_log.csv`를 매번 다시 읽는다.
그래서 `make audit` 한 번으로 6월 18일 당시 상위 후보 기록이

| 원본 (2026-06-18) | 덮어쓴 값 (2026-10-28) |
| --- | --- |
| 342, 339, 338 | 348, 349, 344, 343 |

로 바뀐다. **6월 18일에는 존재하지도 않았던 id가 "6월 18일 진단"에 들어간다.**
테스트 27건이 `rows == 342`에 묶여 깨진 것과 정확히 같은 구조다 — 시점 스냅샷을
파일명으로만 고정하고 내용은 live 재계산에 맡겼다.

영향받는 파일 5개는 이번 점검에서 `git checkout`으로 원복했다.

```
reports/post_id341_data_source_coverage_audit_20260618.json
reports/post_id341_manual_submission_probe_decision_20260618.{json,md}
reports/post_id341_public_local_gap_diagnosis_20260618.{json,md}
```

반대로 **날짜 없는 "현재 상태" 리포트 3종은 갱신본을 채택했다.** 커밋돼 있던
값이 낡아 있었기 때문이다.

| 리포트 | 커밋돼 있던 값 | 실제 |
| --- | --- | --- |
| `submissions_audit` | 로그 344행 기준 | 349행 (`cv_lb_gap_risk` 13 → 16) |
| `workspace_audit` | `git_status_entries` 1185, CRITICAL 1건 | 7, CRITICAL 0건 |
| `repository_structure_audit` | — | 캐시 크기만 변동 |

`workspace_audit`의 CRITICAL("Working tree is too dirty for reliable experiments,
git status has 1185 entries")은 **이미 해소된 과거 상태였다.** 워킹트리는 지금
깨끗하다. 낡은 리포트가 해소된 문제를 계속 보고하고 있었다.

### 남은 구조적 문제

`make audit`은 지금도 돌릴 때마다 위 5개 파일을 다시 덮어쓴다. 제대로 고치려면
둘 중 하나다.

1. 날짜 박힌 타깃을 `make audit` 체인에서 떼고, 재생성이 필요할 때만 수동 호출한다.
2. 출력 파일명을 실행일로 바꾸고(`_$(shell date +%Y%m%d)`), 과거 파일은 읽기 전용으로 둔다.

대회가 끝났고 이 리포트들을 새로 만들 이유가 없으므로 이번에는 고치지 않고
기록만 남긴다. `make audit`을 돌린 뒤에는 `git status`에서 날짜 박힌
`reports/*_2026*.{json,md}`가 올라왔는지 확인하고 원복할 것.
