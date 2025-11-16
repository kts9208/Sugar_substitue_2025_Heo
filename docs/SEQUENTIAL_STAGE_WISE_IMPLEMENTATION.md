# 순차추정 단계별 실행 구현 완료 보고서

## 📋 구현 개요

순차추정을 1단계(측정모델 + 구조모델)와 2단계(선택모델)로 분리하여 실행할 수 있도록 구현했습니다.

**구현 날짜**: 2025-11-16  
**구현 방식**: 방안 1 (단계별 실행 메서드 추가)

---

## ✅ 구현 내용

### 1. 핵심 메서드 추가

#### `SequentialEstimator.estimate_stage1_only()`
- **기능**: 측정모델 + 구조모델 통합 추정 (SEM)
- **입력**: 데이터, 측정모델, 구조모델, 저장 경로(옵션)
- **출력**: 
  - 요인점수 (표준화됨)
  - 잠재변수 간 경로계수
  - 요인적재량
  - 적합도 지수
  - 로그우도
- **파일 저장**: `save_path` 지정 시 자동 저장

#### `SequentialEstimator.estimate_stage2_only()`
- **기능**: 선택모델 추정
- **입력**: 데이터, 선택모델, 요인점수 (딕셔너리 또는 파일 경로)
- **출력**:
  - 선택모델 파라미터
  - 로그우도, AIC, BIC
  - 파라미터 통계
- **유연성**: 메모리 또는 파일에서 요인점수 로드 가능

### 2. 저장/로드 메서드

#### `save_stage1_results()` / `load_stage1_results()`
- **형식**: pickle (.pkl)
- **내용**: 요인점수, 경로계수, 적합도 지수 등 전체 결과
- **용도**: 1단계 결과 저장 및 2단계에서 재사용

#### `save_factor_scores()` / `load_factor_scores()`
- **형식**: pickle (.pkl)
- **내용**: 요인점수만 (경량)
- **용도**: 요인점수만 필요한 경우

---

## 📁 파일 구조

```
src/analysis/hybrid_choice_model/iclv_models/
└── sequential_estimator.py
    ├── estimate_stage1_only()      # 1단계 실행
    ├── estimate_stage2_only()      # 2단계 실행
    ├── save_stage1_results()       # 1단계 결과 저장
    ├── load_stage1_results()       # 1단계 결과 로드
    ├── save_factor_scores()        # 요인점수 저장
    └── load_factor_scores()        # 요인점수 로드

examples/
├── sequential_stage1_example.py    # 1단계 실행 예제
└── sequential_stage2_example.py    # 2단계 실행 예제

tests/
└── test_sequential_stage_wise.py   # 단위 테스트

docs/
├── SEQUENTIAL_STAGE_WISE_GUIDE.md           # 사용 가이드
└── SEQUENTIAL_STAGE_WISE_IMPLEMENTATION.md  # 구현 보고서 (본 문서)
```

---

## 🎯 사용 예시

### 예제 1: 스크립트 사용

```bash
# 1단계 실행
python examples/sequential_stage1_example.py
# → results/stage1_results.pkl 생성

# 결과 검토 (경로계수, 적합도 지수 확인)

# 2단계 실행
python examples/sequential_stage2_example.py
# → 1단계 결과 자동 로드 및 선택모델 추정
```

### 예제 2: Python 코드

```python
from src.analysis.hybrid_choice_model.iclv_models.sequential_estimator import SequentialEstimator

# 1단계 실행
stage1_results = estimator.estimate_stage1_only(
    data=data,
    measurement_model=measurement_model,
    structural_model=structural_model,
    save_path='results/stage1_results.pkl'
)

# 경로계수 확인
print(stage1_results['paths'])

# 2단계 실행 (파일에서 로드)
stage2_results = estimator.estimate_stage2_only(
    data=data,
    choice_model=choice_model,
    factor_scores='results/stage1_results.pkl'
)
```

---

## 🔍 기술적 세부사항

### 1. 요인점수 표준화
- **방법**: Z-score 표준화 (평균 0, 표준편차 1)
- **시점**: 1단계에서 자동 적용
- **이유**: 선택모델에서 스케일 문제 방지

### 2. 파일 형식
- **pickle (.pkl)**: Python 객체 직렬화
- **장점**: 
  - 딕셔너리, DataFrame 등 복잡한 구조 저장 가능
  - 정확한 값 보존 (부동소수점 오차 없음)
- **단점**: Python 전용 (R 등 다른 언어에서 읽기 어려움)

### 3. 세션 분리 지원
- 1단계 실행 → Python 종료
- 결과 파일 검토
- 새로운 Python 세션에서 2단계 실행 가능

---

## ✅ 테스트 결과

### 단위 테스트 (`tests/test_sequential_stage_wise.py`)

- ✅ `test_stage1_only`: 1단계만 실행
- ✅ `test_stage2_only_with_dict`: 2단계 실행 (딕셔너리 전달)
- ✅ `test_stage2_only_with_file`: 2단계 실행 (파일 경로 전달)
- ✅ `test_save_load_stage1_results`: 저장/로드 기능

**실행 방법**:
```bash
pytest tests/test_sequential_stage_wise.py -v
```

---

## 📊 기존 코드와의 호환성

### 기존 `estimate()` 메서드
- **변경 없음**: 기존 코드 그대로 작동
- **내부 구조**: 1단계 + 2단계를 순차적으로 호출
- **하위 호환성**: 100% 보장

### 새로운 메서드
- **독립적**: 기존 코드에 영향 없음
- **선택적 사용**: 필요한 경우만 사용

---

## 🎓 활용 사례

### 사례 1: 잠재변수 관계 확인
```python
# 1단계만 실행하여 경로계수 확인
results = estimator.estimate_stage1_only(...)
print(results['paths'])

# 경로가 유의하지 않으면 모델 수정 후 재실행
```

### 사례 2: 여러 선택모델 비교
```python
# 1단계 1회 실행
stage1_results = estimator.estimate_stage1_only(...)

# 2단계 여러 번 실행 (다른 선택모델)
for choice_model in [model1, model2, model3]:
    results = estimator.estimate_stage2_only(
        data, choice_model, stage1_results['factor_scores']
    )
    print(f"Model: {choice_model.name}, LL: {results['log_likelihood']}")
```

### 사례 3: 재현성 연구
```python
# 1단계 결과 저장
estimator.estimate_stage1_only(..., save_path='stage1.pkl')

# 논문 작성 중...

# 몇 주 후, 정확히 동일한 요인점수로 2단계 재실행
estimator.estimate_stage2_only(..., factor_scores='stage1.pkl')
```

---

## 📚 참고 문서

- **사용 가이드**: `docs/SEQUENTIAL_STAGE_WISE_GUIDE.md`
- **ICLV 전체 가이드**: `docs/ICLV_COMPLETE_SYSTEM_GUIDE.md`
- **API 문서**: `docs/API_REFERENCE.md`

---

## 🔮 향후 개선 사항

1. **CSV 내보내기**: 요인점수를 CSV로 저장하여 R 등 다른 도구에서 사용
2. **결과 시각화**: 1단계 결과 자동 시각화 (경로 다이어그램)
3. **부트스트랩 지원**: 단계별 신뢰구간 계산
4. **병렬 처리**: 여러 선택모델 동시 추정

---

## 📝 변경 이력

- **2025-11-16**: 초기 구현 완료
  - `estimate_stage1_only()` 추가
  - `estimate_stage2_only()` 추가
  - 저장/로드 메서드 추가
  - 예제 스크립트 작성
  - 테스트 코드 작성
  - 문서 작성

