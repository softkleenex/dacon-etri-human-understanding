# ⚖️ 평가 산식 및 지표 (Evaluation Metric)

## 🎯 공식 평가 지표 (Metric)
- Average Log-Loss

## 🧮 산식 (Formula)
$$\text{Score} = \frac{1}{K} \sum_{j=1}^{K} \left[ -\frac{1}{N} \sum_{i=1}^{N} (y_{ij} \log p_{ij} + (1 - y_{ij}) \log(1 - p_{ij})) \right]$$

## 📝 평가 방식 상세 (Evaluation Procedure)
1) **리더보드 평가** : 리더보드 제출 필수 참여 요건.
- Public Score : 전체 테스트 데이터 중 사전 샘플링된 44%
- Private Score : 전체 테스트 데이터 100% (제출마감 후 공개)
- 상위 60팀에게 논문 작성 지원용 Claude API 크레딧 제공
2) **논문 평가 (Accept / Reject)**
- ICTC 2026 Workshop Track(ETRI Human Understanding AI Paper Challenge 부문) 제출된 논문 대상 진행
- 채택된 논문은 반드시 ICTC 2026 학술대회 등록/발표
3) **재현성 검증**
- 논문이 채택된 팀은 코드 및 모델 설명서를 제출해야 함
4) **종합 평가**
- 재현성 검증 통과 논문 대상으로 최종 수상팀(1~5위) 선정 (참신성, 기술적 완성도, 기여도 등 평가)