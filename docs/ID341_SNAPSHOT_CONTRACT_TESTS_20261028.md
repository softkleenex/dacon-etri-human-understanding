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
