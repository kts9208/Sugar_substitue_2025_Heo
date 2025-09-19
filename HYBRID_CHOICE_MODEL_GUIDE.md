# 🎯 하이브리드 선택 모델 (Hybrid Choice Model) 가이드

## 📋 개요

하이브리드 선택 모델은 **이산선택실험(DCE)**과 **구조방정식모델링(SEM)**을 결합한 고급 분석 기법입니다. 설탕 대체재 연구에서 소비자의 선택 행동과 심리적 요인을 동시에 분석할 수 있습니다.

## 🏗️ 시스템 구조

### 📁 모듈 구조
```
src/analysis/hybrid_choice_model/
├── __init__.py                          # 모듈 초기화
├── main_analyzer.py                     # 메인 분석기
├── choice_models/                       # 선택모델들
│   ├── base_choice_model.py            # 기본 클래스
│   ├── choice_model_factory.py         # 팩토리 패턴
│   ├── multinomial_logit_model.py      # MNL 모델
│   ├── random_parameters_logit_model.py # RPL 모델
│   ├── mixed_logit_model.py            # Mixed Logit
│   ├── nested_logit_model.py           # Nested Logit
│   └── multinomial_probit_model.py     # Probit 모델
├── data_integration/                    # 데이터 통합
│   ├── hybrid_data_integrator.py       # 메인 통합기
│   └── __init__.py
└── config/                             # 설정 관리
    ├── hybrid_config.py                # 메인 설정
    ├── estimation_config.py            # 추정 설정
    └── __init__.py
```

## 🎯 지원되는 선택모델

### 1. **다항로짓 모델 (Multinomial Logit, MNL)**
- **특징**: 기본적인 선택모델, 빠른 추정
- **가정**: IIA (Independence of Irrelevant Alternatives)
- **용도**: 기본 분석, 벤치마크

### 2. **확률모수 로짓 모델 (Random Parameters Logit, RPL)**
- **특징**: 개체 이질성 고려, 확률분포 모수
- **장점**: IIA 가정 완화, 현실적 모델링
- **용도**: 개체별 선호 차이 분석

### 3. **혼합로짓 모델 (Mixed Logit)**
- **특징**: 잠재클래스와 확률모수 결합
- **장점**: 세분화된 소비자 그룹 분석
- **용도**: 시장 세분화 연구

### 4. **중첩로짓 모델 (Nested Logit)**
- **특징**: 계층적 선택구조
- **장점**: 대안 간 상관관계 고려
- **용도**: 제품 카테고리 분석

### 5. **다항프로빗 모델 (Multinomial Probit)**
- **특징**: 정규분포 기반 오차항
- **장점**: 유연한 상관구조
- **용도**: 복잡한 선택 패턴 분석

## 🚀 사용법

### 1. 명령행 실행

#### 기본 분석
```bash
# 단일 모델 분석
python main.py --hybrid
python scripts/run_hybrid_choice_analysis.py --model multinomial_logit

# 모델 비교 분석
python main.py --hybrid-compare
python scripts/run_hybrid_choice_analysis.py --compare

# 사용 가능한 모델 목록
python scripts/run_hybrid_choice_analysis.py --list-models
```

#### 고급 옵션
```bash
# RPL 모델 (확률모수 지정)
python scripts/run_hybrid_choice_analysis.py --model random_parameters_logit \
    --random-parameters price sugar_content

# 시뮬레이션 드로우 수 조정
python scripts/run_hybrid_choice_analysis.py --model random_parameters_logit \
    --simulation-draws 2000

# 여러 모델 비교
python scripts/run_hybrid_choice_analysis.py --compare \
    --models multinomial_logit random_parameters_logit mixed_logit
```

### 2. 대화형 메뉴

```bash
python main.py
```

메뉴에서 선택:
- **5. 하이브리드 선택 모델 분석** - 단일 모델 분석
- **6. 하이브리드 모델 비교 분석** - 여러 모델 비교

### 3. Python 코드에서 사용

```python
import pandas as pd
from src.analysis.hybrid_choice_model import (
    run_hybrid_analysis,
    create_default_config,
    create_rpl_config
)

# 데이터 로드
dce_data = pd.read_csv("data/dce_data.csv")
sem_data = pd.read_csv("data/sem_data.csv")

# 기본 분석
config = create_default_config()
result = run_hybrid_analysis(dce_data, sem_data, config)

# RPL 분석
rpl_config = create_rpl_config(['price', 'sugar_content'])
rpl_result = run_hybrid_analysis(dce_data, sem_data, rpl_config)
```

## 📊 데이터 요구사항

### DCE 데이터 형식
```csv
individual_id,choice_set,alternative,choice,price,sugar_content,health_label,brand
1,1,0,1,2000,50,1,A
1,1,1,0,3000,25,0,B
1,1,2,0,2500,75,1,C
...
```

**필수 컬럼:**
- `individual_id`: 개체 식별자
- `choice`: 선택 여부 (0/1)
- `alternative`: 대안 번호
- 기타 속성 변수들

### SEM 데이터 형식
```csv
individual_id,health_concern_1,health_concern_2,...,perceived_benefit_1,...
1,5,4,6,3,5,4,7,6,5,4,3,2,4,5,6,7,3,4,5,6,2,3,4,5
2,6,5,7,4,6,5,8,7,6,5,4,3,5,6,7,8,4,5,6,7,3,4,5,6
...
```

**잠재변수별 관측변수:**
- `health_concern_1~7`: 건강관심도
- `perceived_benefit_1~7`: 지각된유익성
- `purchase_intention_1~4`: 구매의도
- `perceived_price_1~4`: 지각된가격

## 📈 분석 결과

### 1. 데이터 통합 결과
- 총 관측치 수
- DCE/SEM 데이터 크기
- 공통 개체 수
- 통합 성공률

### 2. 측정모델 결과
- 요인적재량 (Factor Loadings)
- 신뢰도 계수 (Reliability)
- 요인점수 (Factor Scores)

### 3. 선택모델 결과
- 모수 추정치 및 표준오차
- 모델 적합도 (Log-likelihood, AIC, BIC)
- 의사결정계수 (Rho-squared)

### 4. 하이브리드 결과
- 통합 모수
- 탄력성 (Elasticity)
- 지불의사액 (Willingness to Pay)

## 🔧 고급 설정

### 1. 확률모수 설정 (RPL)
```python
config = create_rpl_config(
    random_parameters=['price', 'sugar_content'],
    simulation_draws=2000,
    halton_draws=True
)
```

### 2. 중첩구조 설정 (Nested Logit)
```python
config = create_nested_logit_config(
    nesting_structure={
        'healthy': [0, 1],    # 건강한 대안들
        'regular': [2]        # 일반 대안들
    }
)
```

### 3. 추정 방법 설정
```python
config.estimation.method = "simulated_maximum_likelihood"
config.estimation.max_iterations = 2000
config.estimation.convergence_tolerance = 1e-8
```

## ⚠️ 주의사항

### 1. 데이터 품질
- **개체 ID 일치**: DCE와 SEM 데이터의 개체 ID가 일치해야 함
- **결측값 처리**: 결측값은 자동으로 처리되지만 사전 확인 권장
- **데이터 크기**: 최소 100개 이상의 개체 권장

### 2. 모델 선택
- **MNL**: 빠른 분석, 기본 벤치마크
- **RPL**: 개체 이질성이 중요한 경우
- **Mixed Logit**: 시장 세분화가 목적인 경우
- **Nested Logit**: 제품 계층이 명확한 경우

### 3. 수렴 문제
- **초기값**: 다양한 초기값으로 시도
- **반복 횟수**: 충분한 반복 횟수 설정
- **수렴 기준**: 적절한 수렴 기준 설정

## 🎯 실제 사용 예시

### 설탕 대체재 연구에서의 활용

1. **DCE 설계**: 가격, 설탕 함량, 건강 라벨, 브랜드
2. **SEM 측정**: 건강관심도, 지각된유익성, 구매의도, 지각된가격
3. **하이브리드 분석**: 심리적 요인이 선택에 미치는 영향 분석
4. **정책 시뮬레이션**: 건강 라벨 정책의 효과 예측

### 기대 효과

- **통합적 이해**: 선택 행동과 심리적 요인의 동시 분석
- **정확한 예측**: 개체 이질성을 고려한 정확한 수요 예측
- **정책 지원**: 근거 기반 정책 수립 지원
- **시장 세분화**: 소비자 그룹별 맞춤 전략 수립

## 📞 지원 및 문의

- **로그 파일**: `logs/hybrid_choice_analysis.log`
- **결과 디렉토리**: `results/current/hybrid_choice_model/`
- **문서**: 이 가이드 및 코드 내 주석 참조

---

**🎉 축하합니다! 하이브리드 선택 모델이 성공적으로 통합되었습니다!**

이제 DCE와 SEM을 결합한 고급 분석으로 설탕 대체재 연구를 한 단계 더 발전시킬 수 있습니다.
