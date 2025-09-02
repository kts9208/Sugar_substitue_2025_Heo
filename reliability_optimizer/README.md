# 신뢰도 최적화 모듈 (Reliability Optimizer)

## 📋 개요

이 모듈은 기존 신뢰도 분석 결과를 입력받아 AVE(Average Variance Extracted) 기준을 만족하지 못하는 요인의 문항들을 체계적으로 제거하여 크론바흐 알파(Cronbach's Alpha), CR(Composite Reliability), AVE 기준을 모두 만족하는 최적의 문항 조합을 찾는 기능을 제공합니다.

## 🎯 주요 기능

### 1. 자동 문제 요인 식별
- 기존 신뢰도 분석 결과에서 AVE < 0.5인 요인 자동 식별
- 각 요인의 현재 신뢰도 지표 분석

### 2. 체계적인 문항 제거 최적화
- 조합론적 접근을 통한 모든 가능한 문항 제거 조합 시도
- 최소 문항 수(기본값: 3개) 유지하면서 최적화
- 신뢰도 기준을 모두 만족하는 최적해 탐색

### 3. 다중 기준 평가
- **Cronbach's Alpha ≥ 0.7**: 내적 일관성
- **Composite Reliability ≥ 0.7**: 합성 신뢰도
- **AVE ≥ 0.5**: 평균 분산 추출
- **최소 문항 수 ≥ 3**: 요인의 안정성

### 4. 종합 보고서 생성
- 최적화 요약 보고서 (CSV)
- 상세 시도 결과 (CSV)
- 전체 결과 데이터 (JSON)

## 🚀 사용법

### 기본 사용법

```python
from reliability_optimizer import ReliabilityOptimizer

# 1. 최적화기 초기화
optimizer = ReliabilityOptimizer("reliability_analysis_results")

# 2. 기존 결과 로드
optimizer.load_reliability_results()
optimizer.load_raw_data("processed_data/survey_data/nutrition_knowledge.csv")

# 3. 전체 최적화 실행
results = optimizer.optimize_all_problematic_factors(max_removals=10)

# 4. 결과 출력 및 보고서 생성
optimizer.print_optimization_summary(results)
optimizer.generate_optimization_report(results)
```

### 스크립트 실행

```bash
# 전체 최적화 실행
python reliability_optimizer/run_optimization.py
```

## 📁 입력 파일 구조

### 필수 입력 파일들

1. **reliability_analysis_results/reliability_summary.csv**
   ```csv
   Factor,Items,Cronbach_Alpha,Composite_Reliability,AVE,Mean_Loading,Min_Loading,Max_Loading,Alpha_Acceptable,CR_Acceptable,AVE_Acceptable
   nutrition_knowledge,20,0.8101,0.8897,0.4434,0.4737,-0.7411,0.8657,True,True,False
   ```

2. **reliability_analysis_results/factor_loadings.csv**
   ```csv
   Item,Factor,Loading,SE,Z_value,P_value,Significant
   q30,nutrition_knowledge,1.0,-,-,,False
   q31,nutrition_knowledge,0.4658,0.0465,10.0136,0.0,True
   ```

3. **원시 데이터 파일** (예: processed_data/survey_data/nutrition_knowledge.csv)
   ```csv
   no,q30,q31,q32,q33,q34,q35,q36,q37,q38,q39,q40,q41,q42,q43,q44,q45,q46,q47,q48,q49
   1,5,4,5,4,5,2,5,5,4,4,3,4,5,4,5,5,3,1,5,2
   ```

## 📊 출력 결과

### 1. 콘솔 출력
```
🔧 신뢰도 최적화 결과 요약
================================================================================
📊 분석 대상 요인: 1개
✅ 최적화 성공: 1개
❌ 최적화 실패: 0개

📋 요인별 최적화 결과
--------------------------------------------------------------------------------

🔹 nutrition_knowledge
   📈 원본 신뢰도:
      - 문항 수: 20개
      - Cronbach's α: 0.8101
      - CR: 0.8897
      - AVE: 0.4434
   ✨ 최적화 결과:
      - 제거 문항: 7개 (q33, q37, q40, q41, q47, q48, q49)
      - 남은 문항: 13개
      - Cronbach's α: 0.8456
      - CR: 0.9123
      - AVE: 0.5234
      - 모든 기준 충족: ✅
```

### 2. 파일 출력

**reliability_optimization_results/optimization_summary.csv**
- 요인별 최적화 전후 비교 요약

**reliability_optimization_results/optimization_detailed.csv**
- 모든 시도된 조합의 상세 결과

**reliability_optimization_results/optimization_results.json**
- 전체 최적화 결과의 구조화된 데이터

## ⚙️ 설정 옵션

### 신뢰도 기준값 수정

```python
# 기본 기준값
ReliabilityOptimizer.RELIABILITY_THRESHOLDS = {
    'cronbach_alpha': 0.7,
    'composite_reliability': 0.7,
    'ave': 0.5,
    'min_items': 3
}

# 사용자 정의 기준값
optimizer = ReliabilityOptimizer()
optimizer.RELIABILITY_THRESHOLDS['ave'] = 0.6  # AVE 기준을 0.6으로 상향
```

### 최적화 파라미터

```python
# 최대 제거 문항 수 설정
results = optimizer.optimize_all_problematic_factors(max_removals=15)

# 특정 요인만 최적화
result = optimizer.optimize_factor_reliability("nutrition_knowledge", max_removals=10)
```

## 🔧 알고리즘 설명

### 최적화 과정

1. **문제 요인 식별**: AVE < 0.5인 요인들을 자동으로 식별
2. **조합 생성**: 1개부터 max_removals개까지 제거할 문항의 모든 조합 생성
3. **신뢰도 계산**: 각 조합에 대해 Cronbach's α, CR, AVE 계산
4. **기준 검증**: 모든 신뢰도 기준을 만족하는지 확인
5. **최적해 선택**: 기준을 만족하면서 가장 많은 문항을 유지하는 조합 선택

### 점수 계산 공식

```python
score = n_remaining_items + (alpha + cr + ave) / 3
```

- 문항 수가 많을수록 높은 점수
- 신뢰도 지표가 높을수록 높은 점수

## 📈 성능 고려사항

- **계산 복잡도**: O(2^n) - 문항 수가 많을수록 계산 시간 증가
- **권장 최대 문항 수**: 25개 이하
- **최적화 전략**: 적은 수의 문항 제거부터 시작하여 해를 찾으면 중단

## 🔍 예상 결과 (nutrition_knowledge 요인)

현재 nutrition_knowledge 요인의 경우:
- **원본**: 20개 문항, AVE = 0.4434 (기준 미달)
- **예상 최적화 결과**: 13-15개 문항, AVE ≥ 0.5 달성 예상
- **제거 대상 문항**: 요인부하량이 낮거나 음수인 문항들 (q33, q37, q40, q41, q47 등)

## 📞 지원 및 문의

이 모듈은 기존 factor_analysis 패키지와 독립적으로 작동하며, 재사용성과 확장성을 고려하여 설계되었습니다.

- **모듈 위치**: `reliability_optimizer/`
- **실행 스크립트**: `reliability_optimizer/run_optimization.py`
- **결과 저장**: `reliability_optimization_results/`
