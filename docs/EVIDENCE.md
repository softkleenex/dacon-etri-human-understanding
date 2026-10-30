# 근거 자료 (Evidence)

제5회 ETRI 휴먼이해 인공지능 논문경진대회와 ICTC 2026 투고 과정의 1차 자료를 모았다.

**먼저 범위를 분명히 한다.**

| | |
| --- | --- |
| 이 문서가 증명하는 것 | 논문이 ICTC 2026에 **채택**되었고, IEEE 저작권 양식과 PDF eXpress 검증, camera-ready 업로드까지 완료했다는 것 |
| 이 문서가 증명하지 **않는** 것 | **게재.** 저자 등록을 하지 않아 proceedings에 실리지 않았고 IEEE Xplore에도 없다 |

따라서 이 자료를 "IEEE 논문 게재"의 근거로 쓸 수 없다. "채택되었으나 저자 등록
미이행으로 게재되지 않음"이 정확한 서술이다. 경위는
[ICTC/DACON 최종 현황](ICTC_DACON_STATUS_20260916.md)에 있다.

---

## 1. 공개 검증 가능 — 제3자가 직접 확인

| 항목 | 위치 |
| --- | --- |
| 대회 참가·최종 순위 | <https://dacon.io/competitions/official/236690/leaderboard> (로그인 불필요) |
| 대회 개요·규정 | <https://dacon.io/competitions/official/236690/overview/description> |
| 전체 코드·개발 이력 | 이 저장소. 2026-05-06 ~ 2026-09-16, 커밋 55개 |
| 실험 기록 | [`submissions/submission_log.csv`](../submissions/submission_log.csv) 349행, [`reports/`](../reports/) 1,662개, [`notebooks/history/`](../notebooks/history/) 349개 |
| 회고 | <https://softkleenex.github.io/posts/dacon-etri-human-understanding/> |

리더보드 기준 확정 제출은 **ID342** — local CV `0.534000`, public `0.5960832519`, 220위.
해당 행은 `submission_log.csv`의 `id=342`, 생성 노트북과 리포트가 1:1로 대응한다.

## 2. 논문 — 이 저장소에 포함

### 심사본 (2026-07-17 EDAS 업로드)

- [`paper_submission_20260626/review_manuscript_submitted_20260717.pdf`](paper_submission_20260626/review_manuscript_submitted_20260717.pdf)
- [`paper_submission_20260626/paper.tex`](paper_submission_20260626/paper.tex) — LaTeX 원본
- [`paper_submission_20260626/edas_receipt.json`](paper_submission_20260626/edas_receipt.json) — EDAS Paper ID `1571325954`, 업로드본 SHA-256

### 개정본 (리뷰 반영, 2026-09-16 camera-ready)

- [`paper_submission_ictc_camera_ready_20260916/ICTC2026_1571325954_camera_ready.pdf`](paper_submission_ictc_camera_ready_20260916/ICTC2026_1571325954_camera_ready.pdf)
- [`paper_submission_ictc_camera_ready_20260916/reviewer_response.md`](paper_submission_ictc_camera_ready_20260916/reviewer_response.md) — 리뷰어 지적 5건과 각각의 대응

제목: *Anchor-Constrained Feature-Similarity Ensembling for Multimodal Lifelog State Prediction*
저자: Shangjae Lee (Kyungpook National University), 단독 저자. 4페이지.

## 3. 채택 이후 절차 — 이 저장소에 포함

### IEEE 저작권 양식 (eCF)

[`paper_submission_ictc_camera_ready_20260916/IEEE_CopyrightReceipt_1571325954.pdf`](paper_submission_ictc_camera_ready_20260916/IEEE_CopyrightReceipt_1571325954.pdf)

문서 1면에 논문 제목, 저자명, 학회명(`2026 17th International Conference on
Information and Communication Technology Convergence (ICTC)`)이 함께 찍혀 있다.
**채택 사실을 보여주는 단일 문서로는 이게 가장 강하다** — IEEE가 발행했고 학회명과
논문 제목이 한 장에 있다.

서명일 `15-09-2026`.

> **저작권 상태**: 같은 문서 GENERAL TERMS 4항 — *"In the event the above work is not
> accepted and published by the IEEE or is withdrawn by the author(s) before acceptance
> by the IEEE, the foregoing copyright transfer shall be null and void."*
> 이 논문은 IEEE에 게재되지 않았으므로 저작권 양도는 효력이 없고 저작권은 저자에게
> 있다. 이 저장소에 원고를 공개하는 데 제약이 없는 근거다.

### IEEE PDF eXpress 검증

[`paper_submission_ictc_camera_ready_20260916/2026332958_PDFeXpress_certified.pdf`](paper_submission_ictc_camera_ready_20260916/2026332958_PDFeXpress_certified.pdf)

Conference ID `71249X`, PDF eXpress Paper ID `2026332958`. 결과 **Pass**.
검증 사실은 PDF 메타데이터에 남아 있어 파일만으로 확인된다.

```
$ strings 2026332958_PDFeXpress_certified.pdf | grep Creator
/Creator (Certified by IEEE PDFExpress at September 15, 2026 16:40:17)
```

### SHA-256

```
6a07898050061717dcdc381bc20352af191db95c7a6bdfbc766fba01b0bb2a77  IEEE_CopyrightReceipt_1571325954.pdf
221c9ea08a3526f85c1313014698d111d908a67dd2199621a7f021204e445458  2026332958_PDFeXpress_certified.pdf
73836b7caee3a595670bba367763630cd91393b8f9ab7748e6a44c58b3b3c428  ICTC2026_1571325954_camera_ready.pdf
```

## 4. 공개하지 않은 것

| 자료 | 이유 |
| --- | --- |
| EDAS 논문 상태 화면 (`Accepted`) | EDAS 로그인이 필요해 파일로 공개할 수 없다. 요청 시 캡처 제시 |
| 채택 통보 메일, 데이콘 회신 메일 | 담당자 신원·연락처가 포함된 사신. 내용 요지만 [현황 문서](ICTC_DACON_STATUS_20260916.md)에 인용 |
| 등록비 문의 메일 초안, 통화 스크립트 | 수신처 연락처 포함. 저장소 바깥에 둔다 |

---

## 재현성에 대하여

데이콘 재현성 검증에서 **미충족** 판정을 받았다. 이 저장소의 코드로는 원본 데이터에서
최종 `submission.csv`까지 한 경로로 갈 수 없다.

- `reproduce_submission.py` — 저장된 OOF·test 예측 행렬을 재조립한다. SHA-256 일치를
  보장하지만 원본 데이터에서 출발하지 않는다.
- `replay_lineage.py` — 원본에서 출발해 50단계 계보를 완주한다. 다만 2026년 6월 실험
  당시의 scikit-learn/numpy 버전을 고정하지 않아 5단계부터 편차가 누적되고
  최종 log loss가 `+0.000845` 달라진다.

이 결함을 숨기지 않는다. 앙상블 계보의 각 단계가 앞 단계의 *저장된 출력*을 입력으로
받는 구조였고, 그래서 "원본 데이터만으로" 끝까지 가는 경로가 애초에 없었다.
자세한 경위는 [현황 문서](ICTC_DACON_STATUS_20260916.md)에 있다.
