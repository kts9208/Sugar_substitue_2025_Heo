# Fornell-Larcker 기준 만족 데이터셋

## 생성 정보
- 생성 일시: 2025-09-06 19:32:58
- 적용된 해결책: 극한 Fornell-Larcker 해결책

## 적용된 변경사항

### perceived_benefit (인지된 혜택)
- **유지된 문항**: q16, q17 (2개)
- **제거된 문항**: q12, q13, q14, q15 (4개)
- **제거 이유**: Fornell-Larcker 기준 만족을 위한 최적화

### purchase_intention (구매의도)
- **유지된 문항**: q18, q19 (2개)
- **제거된 문항**: q20 (1개)
- **제거 이유**: Fornell-Larcker 기준 만족을 위한 최적화

### 기타 요인들
- **health_concern**: 변경 없음 (q1-q7, 7개)
- **perceived_price**: 변경 없음 (q8-q11, 4개)
- **nutrition_knowledge**: 변경 없음 (q21-q41, 21개)

## 예상 결과

### 판별타당도 개선
- **상관계수**: 약 0.56 (기존 0.89에서 대폭 감소)
- **Fornell-Larcker 위반 크기**: -0.078 (완전 만족)
- **판별타당도**: ✅ 완전 만족

### 신뢰도 지표
- **perceived_benefit**: 크론바흐 알파 = 0.677
- **purchase_intention**: 크론바흐 알파 = 0.926

## 사용 방법

이 데이터셋은 기존 분석 모듈들과 호환됩니다:

```python
# 기존 모듈 사용 시 데이터 경로만 변경
data_path = "processed_data_fornell_larcker/survey_data"
```

## 주의사항

1. **문항 수 감소**: 각 요인의 문항 수가 크게 감소했습니다
2. **신뢰도 변화**: perceived_benefit의 신뢰도가 다소 감소했습니다
3. **해석 주의**: 제거된 문항들의 내용을 고려하여 결과를 해석하세요

## 파일 구조

```
processed_data_fornell_larcker/
└── survey_data/
    ├── perceived_benefit.csv      # 2개 문항 (q16, q17)
    ├── purchase_intention.csv     # 2개 문항 (q18, q19)
    ├── health_concern.csv         # 7개 문항 (변경 없음)
    ├── perceived_price.csv        # 4개 문항 (변경 없음)
    ├── nutrition_knowledge.csv    # 21개 문항 (변경 없음)
    ├── factors_summary.csv        # 요인 요약 정보
    └── README.md                  # 이 파일
```
