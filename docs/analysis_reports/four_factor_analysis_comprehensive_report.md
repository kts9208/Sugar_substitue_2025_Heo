# 4개 요인 분석 종합 보고서

## 개요
본 보고서는 설탕 대체재 연구에서 구매의도를 제외한 4개 핵심 요인에 대한 경로분석과 조절효과 분석 결과를 종합적으로 정리한 것입니다.

**분석 대상 요인:**
1. health_concern (건강 관심)
2. perceived_benefit (인지된 혜택)
3. perceived_price (인지된 가격)
4. nutrition_knowledge (영양 지식)

**분석 일시:** 2025-09-16 19:45:00

---

## 1. 경로분석 결과

### 1.1 전체 분석 개요
- **총 가능한 경로:** 12개 (4×3)
- **유의한 경로:** 11개 (91.7%)
- **분석 방법:** 구조방정식 모델링 (SEM)
- **표본 크기:** 300명

### 1.2 유의한 경로 상세

#### 매우 강한 유의성 (p < 0.001)
1. **perceived_price → perceived_benefit** (β = 4.7359, p < 0.001)
2. **nutrition_knowledge → perceived_benefit** (β = -24.6032, p < 0.001)
3. **health_concern → perceived_price** (β = -7.2203, p < 0.001)
4. **perceived_benefit → perceived_price** (β = 5.7010, p < 0.001)
5. **nutrition_knowledge → perceived_price** (β = 6.2672, p < 0.001)
6. **health_concern → nutrition_knowledge** (β = 11.4337, p < 0.001)
7. **perceived_benefit → nutrition_knowledge** (β = -2.5774, p < 0.001)
8. **perceived_price → nutrition_knowledge** (β = -4.9383, p < 0.001)
9. **perceived_benefit → health_concern** (β = 17.8416, p < 0.001)
10. **perceived_price → health_concern** (β = -8.9435, p < 0.001)

#### 유의한 경로 (p < 0.05)
11. **nutrition_knowledge → health_concern** (β = -3.4611, p = 0.019)

### 1.3 요인별 영향 관계 분석

#### health_concern (건강 관심)
- **영향을 주는 관계:** 2개
  - perceived_price에 부정적 영향 (-7.2203)
  - nutrition_knowledge에 긍정적 영향 (11.4337)
- **영향을 받는 관계:** 3개
  - perceived_benefit으로부터 강한 긍정적 영향 (17.8416)
  - perceived_price로부터 부정적 영향 (-8.9435)
  - nutrition_knowledge로부터 약한 부정적 영향 (-3.4611)

#### perceived_benefit (인지된 혜택)
- **영향을 주는 관계:** 3개
  - health_concern에 강한 긍정적 영향 (17.8416)
  - perceived_price에 긍정적 영향 (5.7010)
  - nutrition_knowledge에 부정적 영향 (-2.5774)
- **영향을 받는 관계:** 2개
  - perceived_price로부터 긍정적 영향 (4.7359)
  - nutrition_knowledge로부터 강한 부정적 영향 (-24.6032)

#### perceived_price (인지된 가격)
- **영향을 주는 관계:** 3개
  - perceived_benefit에 긍정적 영향 (4.7359)
  - health_concern에 부정적 영향 (-8.9435)
  - nutrition_knowledge에 부정적 영향 (-4.9383)
- **영향을 받는 관계:** 3개
  - health_concern으로부터 부정적 영향 (-7.2203)
  - perceived_benefit으로부터 긍정적 영향 (5.7010)
  - nutrition_knowledge로부터 긍정적 영향 (6.2672)

#### nutrition_knowledge (영양 지식)
- **영향을 주는 관계:** 3개
  - perceived_benefit에 강한 부정적 영향 (-24.6032)
  - perceived_price에 긍정적 영향 (6.2672)
  - health_concern에 약한 부정적 영향 (-3.4611)
- **영향을 받는 관계:** 3개
  - health_concern으로부터 강한 긍정적 영향 (11.4337)
  - perceived_benefit으로부터 부정적 영향 (-2.5774)
  - perceived_price로부터 부정적 영향 (-4.9383)

---

## 2. 조절효과 분석 결과

### 2.1 전체 분석 개요
- **총 분석 조합:** 24개 (4×3×2)
- **성공한 분석:** 24개 (100.0%)
- **유의한 조절효과:** 0개 (0.0%)
- **분석 방법:** 상호작용항을 포함한 구조방정식 모델링

### 2.2 조절효과 분석 상세 결과

모든 24개 조절효과 조합에서 통계적으로 유의한 조절효과가 발견되지 않았습니다 (모든 p > 0.05).

#### 주요 조절효과 분석 결과 (p값 기준 상위 10개)
1. **perceived_benefit × nutrition_knowledge → health_concern** (p = 0.154)
2. **health_concern × perceived_price → perceived_benefit** (p = 0.231)
3. **perceived_price × nutrition_knowledge → perceived_benefit** (p = 0.415)
4. **nutrition_knowledge × perceived_benefit → perceived_price** (p = 0.290)
5. **health_concern × perceived_benefit → nutrition_knowledge** (p = 0.465)
6. **perceived_benefit × perceived_price → nutrition_knowledge** (p = 0.497)

### 2.3 요인별 조절효과 분석

#### health_concern (종속변수)
- 총 분석: 6개
- 유의한 조절효과: 0개

#### perceived_benefit (종속변수)
- 총 분석: 6개
- 유의한 조절효과: 0개

#### perceived_price (종속변수)
- 총 분석: 6개
- 유의한 조절효과: 0개

#### nutrition_knowledge (종속변수)
- 총 분석: 6개
- 유의한 조절효과: 0개

---

## 3. 종합 분석 및 해석

### 3.1 주요 발견사항

#### 경로분석
1. **높은 상호연관성:** 4개 요인 간 대부분의 경로가 유의함 (91.7%)
2. **복잡한 상호작용:** 모든 요인이 서로 영향을 주고받는 복잡한 네트워크 구조
3. **강한 영향 관계:** 특히 nutrition_knowledge와 perceived_benefit 간 강한 부정적 관계

#### 조절효과 분석
1. **조절효과 부재:** 4개 요인 간 유의한 조절효과가 전혀 발견되지 않음
2. **직접적 관계 우세:** 요인 간 관계가 주로 직접적이며 조건부적이지 않음

### 3.2 이론적 함의

#### 경로분석 결과의 함의
1. **시스템적 접근 필요:** 4개 요인이 독립적이 아닌 상호의존적 시스템으로 작동
2. **영양 지식의 핵심 역할:** nutrition_knowledge가 다른 요인들에 강한 영향을 미침
3. **인지된 혜택의 중재 역할:** perceived_benefit이 다른 요인들을 연결하는 중재 역할

#### 조절효과 부재의 함의
1. **선형적 관계:** 요인 간 관계가 주로 선형적이며 조건부적이지 않음
2. **일관된 영향:** 한 요인의 영향이 다른 요인의 수준에 관계없이 일관됨
3. **단순한 인과구조:** 복잡한 조건부 관계보다는 직접적 인과관계가 우세

### 3.3 실무적 시사점

1. **통합적 마케팅 전략:** 4개 요인을 개별적이 아닌 통합적으로 접근해야 함
2. **영양 교육 우선순위:** nutrition_knowledge 향상이 다른 요인들에 파급효과를 가져올 수 있음
3. **직접적 개입 효과적:** 조절효과가 없으므로 각 요인에 대한 직접적 개입이 효과적

---

## 4. 연구의 한계 및 향후 연구 방향

### 4.1 연구의 한계
1. **표본 크기:** 300명의 표본으로 조절효과 탐지에 제한적일 수 있음
2. **측정 방법:** 자기보고식 설문의 한계
3. **횡단적 연구:** 인과관계 추론의 한계

### 4.2 향후 연구 방향
1. **종단적 연구:** 시간에 따른 요인 간 관계 변화 탐구
2. **실험적 설계:** 인과관계 확립을 위한 실험적 접근
3. **조절변수 탐색:** 다른 잠재적 조절변수 탐색
4. **비선형 관계:** 비선형적 조절효과 가능성 탐구

---

## 5. 결론

본 연구는 설탕 대체재 관련 4개 핵심 요인 간의 관계를 경로분석과 조절효과 분석을 통해 종합적으로 탐구했습니다.

**주요 결론:**
1. 4개 요인 간 강한 직접적 관계가 존재함 (11/12 경로 유의)
2. 조절효과는 발견되지 않아 관계가 주로 직접적이고 일관됨
3. nutrition_knowledge가 시스템의 핵심 요인으로 작용함
4. 실무적으로는 통합적 접근과 직접적 개입이 효과적일 것으로 예상됨

이러한 결과는 설탕 대체재 마케팅 전략 수립과 소비자 행동 이해에 중요한 통찰을 제공합니다.
