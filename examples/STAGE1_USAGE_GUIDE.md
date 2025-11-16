# 1단계 추정 사용 가이드

## 📋 개요

1단계 추정은 **측정모델(CFA) + 구조모델(SEM)**을 추정하여 잠재변수 간 관계를 확인합니다.

## 📁 파일 구조

```
examples/
├── sequential_stage1.py          # ⭐ 1단계 추정 (통합 버전)
├── sequential_cfa_only_example.py # CFA만 추정 (상관관계 확인)
└── STAGE1_USAGE_GUIDE.md         # 이 파일
```

---

## 🚀 빠른 시작

### 1. 경로 설정

`examples/sequential_stage1.py` 파일을 열고 `PATHS` 딕셔너리를 수정합니다.

```python
PATHS = {
    'HC->PB': True,   # 건강관심도 → 건강유익성
    'HC->PP': False,  # 건강관심도 → 가격수준
    'HC->PI': False,  # 건강관심도 → 구매의도
    'PB->PI': True,   # 건강유익성 → 구매의도
    'PP->PI': False,  # 가격수준 → 구매의도
    'NK->PI': False,  # 영양지식 → 구매의도
}
```

### 2. 실행

```bash
python examples/sequential_stage1.py
```

### 3. 결과 확인

결과 파일이 `results/sequential_stage_wise/` 폴더에 저장됩니다:
- `stage1_HC-PB_PB-PI_results.pkl` (2단계에서 사용)
- `stage1_HC-PB_PB-PI_results_paths.csv` (경로계수)
- `stage1_HC-PB_PB-PI_results_loadings.csv` (요인적재량)
- `stage1_HC-PB_PB-PI_results_fit_indices.csv` (적합도 지수)
- `stage1_HC-PB_PB-PI_results_factor_scores.csv` (요인점수)

---

## 📊 주요 모델 예시

### 1. Base Model (HC→PB→PI)

**가장 간결한 모델**

```python
PATHS = {
    'HC->PB': True,
    'HC->PP': False,
    'HC->PI': False,
    'PB->PI': True,
    'PP->PI': False,
    'NK->PI': False,
}
```

**결과 파일명:** `stage1_HC-PB_PB-PI_results.*`

---

### 2. 확장 모델 (HC→PB→PI + HC→PP→PI)

**현재 논문에서 사용 중인 모델**

```python
PATHS = {
    'HC->PB': True,
    'HC->PP': True,
    'HC->PI': False,
    'PB->PI': True,
    'PP->PI': True,
    'NK->PI': False,
}
```

**결과 파일명:** `stage1_HC-PB_HC-PP_PB-PI_PP-PI_results.*`

---

### 3. 완전 모델 (모든 경로)

**탐색적 분석용**

```python
PATHS = {
    'HC->PB': True,
    'HC->PP': True,
    'HC->PI': True,
    'PB->PI': True,
    'PP->PI': True,
    'NK->PI': True,
}
```

**결과 파일명:** `stage1_HC-PB_HC-PI_HC-PP_NK-PI_PB-PI_PP-PI_results.*`

---

## 🔧 고급 설정

### 수정지수 계산

모델 개선을 위한 경로 추가 제안을 받으려면:

```python
CALCULATE_MODIFICATION_INDICES = True
```

---

## 📖 잠재변수 약어

| 약어 | 전체 이름 | 한글 이름 |
|------|----------|----------|
| HC | health_concern | 건강관심도 |
| PB | perceived_benefit | 건강유익성 |
| PP | perceived_price | 가격수준 |
| NK | nutrition_knowledge | 영양지식 |
| PI | purchase_intention | 구매의도 |

---

## 🎯 다음 단계

1단계 추정 완료 후 2단계 선택모델을 추정하려면:

1. `examples/sequential_stage2_with_extended_model.py` 열기
2. `STAGE1_RESULT_FILE` 설정:
   ```python
   STAGE1_RESULT_FILE = "stage1_HC-PB_PB-PI_results.pkl"
   ```
3. 실행:
   ```bash
   python examples/sequential_stage2_with_extended_model.py
   ```

---

## ❓ FAQ

**Q: 경로가 없는 모델(CFA만)을 추정하려면?**

A: 모든 경로를 `False`로 설정하거나, `sequential_cfa_only_example.py`를 사용하세요.

**Q: 파일명이 너무 길어요.**

A: 경로가 많을수록 파일명이 길어집니다. 필요한 경로만 활성화하세요.

**Q: 2단계에서 어떤 1단계 결과를 사용해야 하나요?**

A: 일반적으로 Base Model (HC→PB→PI)을 권장합니다. 모델이 간결하고 해석이 명확합니다.

