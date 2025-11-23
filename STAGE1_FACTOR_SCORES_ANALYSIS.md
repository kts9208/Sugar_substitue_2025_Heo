# 순차추정 2단계에서 사용하는 1단계 요인점수 분석

**분석 날짜:** 2025-11-23  
**분석자:** ICLV Team

---

## 📋 요약

순차추정 2단계 코드(`examples/sequential_stage2_with_extended_model.py`)는 1단계에서 생성된 요인점수를 사용하여 선택모델을 추정합니다. 이 문서는 1단계 요인점수의 구조, 변환 방법, 2단계에서의 사용 방식을 상세히 분석합니다.

---

## 1️⃣ 1단계 결과 파일 구조

### 파일 위치
```
results/sequential_stage_wise/stage1_HC-PB_PB-PI_results.pkl
```

### 파일 내용 (PKL)
```python
{
    'factor_scores': Dict[str, np.ndarray],           # ✅ 변환된 요인점수 (Z-score)
    'original_factor_scores': Dict[str, np.ndarray],  # ✅ 원본 요인점수 (SEM 추출 직후)
    'paths': pd.DataFrame,                            # 잠재변수 간 경로계수
    'loadings': pd.DataFrame,                         # 요인적재량
    'fit_indices': Dict,                              # 적합도 지수
    'log_likelihood': float,                          # 로그우도
    'measurement_results': Dict,                      # 측정모델 결과
    'structural_results': Dict,                       # 구조모델 결과
    'version': str                                    # 버전 정보
}
```

### 요인점수 변수 (5개)
1. `health_concern` (HC) - 건강관심도
2. `perceived_benefit` (PB) - 건강유익성
3. `perceived_price` (PP) - 가격수준
4. `nutrition_knowledge` (NK) - 영양지식
5. `purchase_intention` (PI) - 구매의도

**Shape:** 각 변수당 `(326,)` - 326명의 개인

---

## 2️⃣ 요인점수 통계

### 변환된 요인점수 (Z-score 표준화)

| 변수명 | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| health_concern | 0.000000 | 1.000000 | -4.042472 | 1.908734 |
| perceived_benefit | 0.000000 | 1.000000 | -3.081095 | 2.740968 |
| perceived_price | 0.000000 | 1.000000 | -3.127351 | 2.190839 |
| nutrition_knowledge | 0.000000 | 1.000000 | -3.193505 | 2.345300 |
| purchase_intention | 0.000000 | 1.000000 | -2.692580 | 1.681654 |

**특징:**
- ✅ 평균 = 0 (정확히 0)
- ✅ 표준편차 = 1 (정확히 1)
- ✅ Z-score 표준화 완료

### 원본 요인점수 (SEM 추출 직후)

| 변수명 | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| health_concern | 3.953770 | 0.714932 | 1.063677 | 5.318386 |
| perceived_benefit | 3.493666 | 0.682137 | 1.391938 | 5.363381 |
| perceived_price | 4.065249 | 0.714625 | 1.830367 | 5.630876 |
| nutrition_knowledge | 2.775007 | 0.615173 | 0.810448 | 4.217772 |
| purchase_intention | 3.509722 | 0.926992 | 1.013720 | 5.068602 |

**특징:**
- ✅ 원본 스케일 유지 (Likert 5점 척도 범위)
- ✅ 평균 약 2.8~4.1
- ✅ 표준편차 약 0.6~0.9

---

## 3️⃣ 요인점수 변환 방법

### Z-score 표준화 공식
```python
z = (x - mean(x)) / std(x)
```

### 검증 결과
```
변수: health_concern
  원본 평균: 3.953770
  원본 표준편차: 0.714932
  변환 후 평균: 0.000000
  변환 후 표준편차: 1.000000
  수동 Z-score와 일치: True ✅
```

### 변환 코드 위치
<augment_code_snippet path="src/analysis/hybrid_choice_model/iclv_models/sequential_estimator.py" mode="EXCERPT">
````python
def _standardize_factor_scores(self, factor_scores: Dict[str, np.ndarray],
                                method: str = 'zscore') -> Dict[str, np.ndarray]:
    """
    요인점수 표준화 또는 중심화
    
    - method='zscore': Z-score 표준화 (평균 0, 표준편차 1) - 기본값
    - method='center': 중심화 (평균 0, 표준편차는 원본 유지)
    """
````
</augment_code_snippet>

---

## 4️⃣ 2단계에서 요인점수 사용 방식

### 코드 흐름

<augment_code_snippet path="examples/sequential_stage2_with_extended_model.py" mode="EXCERPT">
````python
# 1. 1단계 결과 파일 경로 지정
STAGE1_RESULT_FILE = "stage1_HC-PB_PB-PI_results.pkl"
stage1_path = project_root / "results" / "sequential_stage_wise" / STAGE1_RESULT_FILE

# 2. 2단계 추정 실행
results = estimator.estimate_stage2_only(
    data=data,
    choice_model=choice_model,
    factor_scores=str(stage1_path)  # 파일 경로 전달
)
````
</augment_code_snippet>

### 내부 처리 과정

<augment_code_snippet path="src/analysis/hybrid_choice_model/iclv_models/sequential_estimator.py" mode="EXCERPT">
````python
# 요인점수 로드 (파일 경로인 경우)
if isinstance(factor_scores, str):
    loaded_results = self.load_stage1_results(factor_scores)
    
    # ✅ 원본 요인점수가 있으면 사용, 없으면 변환된 요인점수 사용
    if 'original_factor_scores' in loaded_results:
        original_factor_scores = loaded_results['original_factor_scores']
        
        # 현재 설정에 맞게 재변환
        self.factor_scores = self._standardize_factor_scores(
            original_factor_scores,
            method=self.standardization_method
        )
    else:
        self.factor_scores = loaded_results['factor_scores']
````
</augment_code_snippet>

### 처리 단계

1. **파일 로드**
   - `stage1_HC-PB_PB-PI_results.pkl` 로드
   
2. **원본 요인점수 확인**
   - `'original_factor_scores'` 키 존재 여부 확인
   
3. **재변환 (존재하는 경우)**
   - 원본 요인점수를 현재 설정(`STANDARDIZATION_METHOD`)에 맞게 재변환
   - 1단계: `zscore`, 2단계: `center` 등 다른 방법 사용 가능
   
4. **그대로 사용 (없는 경우)**
   - 저장된 `'factor_scores'` 그대로 사용
   - 경고 메시지 출력

---

## 5️⃣ 요인점수 확장 (개인 → 선택 상황)

### 데이터 구조 변환

**1단계 요인점수:**
- Shape: `(326,)` - 326명의 개인별 요인점수

**2단계 선택 데이터:**
- Shape: `(n_rows,)` - 개인 × 선택 세트 × 대안
- 예: 326명 × 8개 선택 세트 × 3개 대안 = 7,824행

### 확장 방법

<augment_code_snippet path="src/analysis/hybrid_choice_model/iclv_models/choice_equations.py" mode="EXCERPT">
````python
# respondent_id 기준으로 요인점수 매핑
unique_ids = data['respondent_id'].unique()

lv_expanded = {}
for lv_name, scores in factor_scores.items():
    # 각 행의 respondent_id에 해당하는 요인점수 할당
    id_to_score = {unique_ids[i]: scores[i] for i in range(len(unique_ids))}
    expanded = np.array([id_to_score[rid] for rid in data['respondent_id']])
    lv_expanded[lv_name] = expanded
````
</augment_code_snippet>

### 확장 예시

```
개인 1의 PI 요인점수: -0.1469
  → 개인 1의 모든 선택 상황 (8개 세트 × 3개 대안 = 24행)에 -0.1469 할당

개인 2의 PI 요인점수: -0.1401
  → 개인 2의 모든 선택 상황 (24행)에 -0.1401 할당
```

**결과:**
- 확장 전: `(326,)`
- 확장 후: `(7824,)` (예시)
- 통계량 유지: Mean, Std 동일

---

## 6️⃣ CSV 파일 저장

### 파일 위치
```
results/sequential_stage_wise/stage1_HC-PB_PB-PI_results_factor_scores.csv
```

### 파일 구조
```csv
observation_id,health_concern,perceived_benefit,perceived_price,nutrition_knowledge,purchase_intention
0,-0.30690262,0.14214676,0.61494116,1.05750195,-0.14685583
1,-0.28737186,0.12533933,0.24843808,-1.23403744,-0.14008932
2,-0.07036534,1.69786274,-1.55145343,0.69901405,1.68165395
...
```

**특징:**
- ✅ 변환된 요인점수 (Z-score) 저장
- ✅ PKL 파일과 일치
- ✅ 326행 × 6열 (ID + 5개 LV)

---

## 7️⃣ 장점 및 특징

### ✅ 원본 요인점수 보존
- `original_factor_scores` 키에 SEM 추출 직후 요인점수 저장
- 1단계와 2단계에서 다른 변환 방법 사용 가능
- 유연성 확보

### ✅ 자동 재변환
- 2단계에서 현재 설정에 맞게 자동 재변환
- `STANDARDIZATION_METHOD` 변경 시 자동 적용

### ✅ 하위 호환성
- 원본 요인점수가 없는 경우에도 작동
- 경고 메시지로 사용자에게 알림

### ✅ 부트스트랩 안전
- `respondent_id` 기준 매핑으로 부트스트랩 샘플링 시에도 안전

---

## 8️⃣ 사용 예시

### 1단계 실행 (요인점수 생성)
```python
# examples/sequential_stage1.py
PATHS = {
    'HC->PB': True,
    'PB->PI': True,
}
STANDARDIZATION_METHOD = 'zscore'  # Z-score 표준화

# 실행 → stage1_HC-PB_PB-PI_results.pkl 생성
```

### 2단계 실행 (요인점수 사용)
```python
# examples/sequential_stage2_with_extended_model.py
STAGE1_RESULT_FILE = "stage1_HC-PB_PB-PI_results.pkl"
STANDARDIZATION_METHOD = 'zscore'  # 1단계와 동일

MAIN_LVS = ['purchase_intention', 'nutrition_knowledge']
LV_ATTRIBUTE_INTERACTIONS = [
    ('purchase_intention', 'health_label'),
    ('nutrition_knowledge', 'price')
]

# 실행 → 자동으로 요인점수 로드 및 재변환
```

### 다른 변환 방법 사용
```python
# 2단계에서 중심화 사용
STANDARDIZATION_METHOD = 'center'  # 평균 0, 표준편차는 원본 유지

# 원본 요인점수에서 자동 재변환됨
```

---

## 9️⃣ 요인점수 흐름도

```
[1단계: SEM 추정]
    ↓
[요인점수 추출 (원본 스케일)]
    ↓
[Z-score 표준화]
    ↓
[PKL 저장]
  - factor_scores (변환됨)
  - original_factor_scores (원본)
    ↓
[2단계: 선택모델 추정]
    ↓
[PKL 로드]
    ↓
[원본 요인점수 확인]
    ↓
[현재 설정에 맞게 재변환]
    ↓
[개인 → 선택 상황 확장]
    ↓
[선택모델 추정]
```

---

## 🔟 주요 코드 위치

| 기능 | 파일 | 함수/메서드 |
|------|------|------------|
| 요인점수 추출 | `sem_estimator.py` | `_extract_factor_scores()` |
| 요인점수 변환 | `sequential_estimator.py` | `_standardize_factor_scores()` |
| 1단계 결과 저장 | `sequential_estimator.py` | `estimate_stage1_only()` |
| 1단계 결과 로드 | `sequential_estimator.py` | `load_stage1_results()` |
| 2단계 추정 | `sequential_estimator.py` | `estimate_stage2_only()` |
| 요인점수 확장 | `choice_equations.py` | `fit()` 메서드 내부 |

---

## 📊 결론

**✅ 순차추정 2단계는 1단계 요인점수를 다음과 같이 사용합니다:**

1. **PKL 파일 로드:** `stage1_*.pkl` 파일에서 요인점수 로드
2. **원본 요인점수 사용:** `original_factor_scores` 존재 시 재변환
3. **자동 재변환:** 현재 설정(`STANDARDIZATION_METHOD`)에 맞게 변환
4. **확장:** 개인별 요인점수를 선택 상황별로 확장 (`respondent_id` 기준)
5. **선택모델 추정:** 확장된 요인점수를 사용하여 선택모델 추정

**✅ 주요 특징:**
- 원본 요인점수 보존으로 유연성 확보
- 자동 재변환으로 편리성 제공
- 부트스트랩 안전한 확장 방법
- Z-score 표준화 기본값 (평균 0, 표준편차 1)

---

**분석 완료!** 🎉

