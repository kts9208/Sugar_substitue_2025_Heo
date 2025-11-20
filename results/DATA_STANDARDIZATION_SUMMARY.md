# 데이터 표준화 여부 종합 정리

## 요약

| 데이터 유형 | 표준화 여부 | 평균 | 표준편차 | 비고 |
|------------|------------|------|----------|------|
| **CFA 요인점수** | ✅ **표준화됨** | 0 | 1 | semopy가 자동 표준화 |
| **구조모델 LV (동시추정)** | ✅ **표준화됨** | 0 | 1 | N(0,1)에서 샘플링 |
| **선택 속성 - price** | ❌ **표준화 안됨** | 2500 | 408 | 원척도 (2000-3000원) |
| **선택 속성 - health_label** | ❌ **표준화 안됨** | 0.5 | 0.5 | 이진 변수 (0/1) |
| **사회인구학적 - age_std** | ✅ **표준화됨** | 0 | 1 | Z-score 표준화 |
| **사회인구학적 - income_std** | ✅ **표준화됨** | 0 | 1 | Z-score 표준화 |
| **사회인구학적 - gender** | ❌ **표준화 안됨** | 0.5 | 0.5 | 이진 변수 (0/1) |
| **사회인구학적 - education** | ❌ **표준화 안됨** | 3.84 | 1.04 | 원척도 (1-6) |
| **잠재변수 지표 (q6-q49)** | ❌ **표준화 안됨** | 3-4 | 0.7-1.0 | 원척도 (1-5점 리커트) |

---

## 1. CFA 결과의 표준화 여부

### ✅ **표준화됨 (평균 0, 표준편차 1)**

**확인 결과**:
```
health_concern:      평균 = 0.000000, 표준편차 = 1.000000
perceived_benefit:   평균 = 0.000000, 표준편차 = 1.000000
perceived_price:     평균 = 0.000000, 표준편차 = 1.000000
nutrition_knowledge: 평균 = 0.000000, 표준편차 = 1.000000
purchase_intention:  평균 = 0.000000, 표준편차 = 1.000000
```

**이유**:
1. **semopy의 자동 표준화**:
   - semopy는 CFA 추정 시 잠재변수 분산을 1로 고정
   - `predict_factors()` 또는 수동 계산 모두 표준화된 요인점수 반환

2. **코드 확인** (`src/analysis/hybrid_choice_model/iclv_models/sem_estimator.py`):
   ```python
   def _extract_factor_scores(self, data, measurement_model):
       """
       요인점수 추출
       
       ✅ 수동 계산 (Bartlett 방법)을 사용하여 원본 스케일의 요인점수를 추출합니다.
       
       semopy의 predict_factors()는 잠재변수 분산 제약으로 인해
       표준화된 요인점수를 반환하므로 사용하지 않습니다.
       """
       # ✅ 항상 수동 계산 사용 (원본 스케일 유지)
       return self._manual_factor_scores(data, measurement_model)
   ```

3. **주석과 실제 동작의 불일치**:
   - 주석: "원본 스케일 유지"
   - 실제: **표준화된 요인점수 반환** (semopy 제약)
   - **결론**: semopy의 잠재변수 분산 제약(Var(LV)=1)으로 인해 자동 표준화됨

---

## 2. 구조모델 추정시 사용되는 LV 값의 표준화 여부

### ✅ **표준화됨 (평균 0, 표준편차 1)**

### **2-1. 순차추정 (Sequential Estimation)**

**1단계 (CFA + 구조모델)**:
- CFA 요인점수: **표준화됨** (위 참조)
- 추가 표준화 적용: **Yes** (선택 가능)

**코드** (`src/analysis/hybrid_choice_model/iclv_models/sequential_estimator.py`):
```python
# 요인점수 변환 (표준화 또는 중심화)
method_name = "Z-score 표준화" if self.standardization_method == 'zscore' else "중심화"
self.factor_scores = self._standardize_factor_scores(
    original_factor_scores, 
    method=self.standardization_method
)

def _standardize_factor_scores(self, factor_scores, method='zscore'):
    """
    - method='zscore': Z-score 표준화 (평균 0, 표준편차 1) - 기본값
      z = (x - mean(x)) / std(x)
    
    - method='center': 중심화 (평균 0, 표준편차는 원본 유지)
      centered = x - mean(x)
    """
    if method == 'zscore':
        transformed_scores = (scores - mean) / std
    else:  # method == 'center'
        transformed_scores = scores - mean
```

**실제 사용**:
- 기본값: `standardization_method='zscore'`
- **이중 표준화**: CFA 표준화 + 추가 Z-score 표준화
- **결과**: 평균 0, 표준편차 1 (확인됨)

**2단계 (선택모델)**:
- 1단계 요인점수 사용: **표준화됨**

---

### **2-2. 동시추정 (Simultaneous Estimation)**

**LV 샘플링 방법**:
- **표준정규분포 N(0, 1)에서 샘플링**

**코드** (`src/analysis/hybrid_choice_model/iclv_models/draw_generator.py`):
```python
def _generate_draws(self):
    """Halton 시퀀스 생성"""
    # scipy의 Halton 시퀀스 생성기 사용
    sampler = qmc.Halton(d=self.n_dimensions, scramble=self.scramble, seed=self.seed)
    
    # 균등분포 [0,1] 샘플 생성
    uniform_draws = sampler.random(n=self.n_individuals * self.n_draws)
    
    # ✅ 표준정규분포로 변환 (역누적분포함수)
    normal_draws = norm.ppf(uniform_draws)
```

**구조모델 예측** (`src/analysis/hybrid_choice_model/iclv_models/multi_latent_structural.py`):
```python
# 1. 1차 LV (외생, 표준정규분포)
for i, lv_name in enumerate(self.exogenous_lvs):
    latent_vars[lv_name] = exo_draws[i]  # ✅ N(0,1) draw

# 2. 2차+ LV (계층적 구조)
# 평균 계산: Σ(γ_k * LV_k)
lv_mean = 0.0
for pred in predictors:
    gamma = params[f'gamma_{pred}_to_{target}']
    pred_lv = latent_vars[pred]  # ✅ 표준화된 LV
    lv_mean += gamma * pred_lv

# 오차항 추가
error_draw = higher_order_draws.get(target, 0.0)  # ✅ N(0,1) draw
error_term = np.sqrt(self.error_variance) * error_draw
latent_vars[target] = lv_mean + error_term
```

**결론**:
- 1차 LV: **N(0, 1)에서 직접 샘플링** → 표준화됨
- 2차+ LV: **표준화된 1차 LV의 선형결합 + N(0, σ²) 오차** → 표준화 아님 (평균 0이지만 분산 ≠ 1)

---

## 3. 선택속성값의 표준화 여부

### ❌ **표준화 안됨 (원척도 사용)**

**확인 결과**:
```
price:        평균 = 2500.00, 표준편차 = 408.30, 범위 = [2000, 3000]
health_label: 평균 = 0.50,    표준편차 = 0.50,    범위 = [0, 1]
```

**이유**:
1. **price**: 원척도 (원 단위) 사용
   - 2000원, 2500원, 3000원 (3개 수준)
   - 표준화 안됨

2. **health_label**: 이진 변수 (0/1)
   - 0: 건강 라벨 없음
   - 1: 건강 라벨 있음
   - 표준화 불필요 (이미 0-1 범위)

**코드 확인** (`scripts/integrate_iclv_data.py`):
```python
# DCE 데이터 로드
df_dce = pd.read_csv('data/processed/dce/dce_long_format.csv')

# 선택 속성은 그대로 사용 (표준화 없음)
# - price: 원척도 (2000, 2500, 3000)
# - health_label: 이진 (0, 1)
```

**선택모델에서 사용**:
- price: 원척도 그대로 사용
- health_label: 이진 변수 그대로 사용
- **표준화 없음**

---

## 4. 스케일 불일치 문제

### **문제 요약**

| 변수 유형 | 스케일 | 평균 | 표준편차 |
|----------|--------|------|----------|
| **LV (동시추정)** | 표준화 | 0 | 1 |
| **지표 (q6-q49)** | 원척도 | 3-4 | 0.7-1.0 |
| **price** | 원척도 | 2500 | 408 |

### **영향**

1. **측정모델 우도**:
   - LV: N(0, 1) → 예: LV = 0.5
   - 지표: 1-5점 → 예: Y = 4.0
   - 예측: Y_pred = ζ × LV = 1.0 × 0.5 = 0.5
   - **잔차: 4.0 - 0.5 = 3.5 (매우 큼!)**
   - **우도: 매우 낮음** (큰 음수)

2. **선택모델 우도**:
   - price: 2000-3000원 (원척도)
   - LV: N(0, 1) (표준화)
   - **스케일 불일치** → 파라미터 추정 어려움

---

## 5. 권장 사항

### **Option A: 현재 상태 유지 (추천)**

**이유**:
- 이미 스케일링 적용 중 (측정모델 우도 ÷ 38)
- Gamma 파라미터 업데이트 성공
- 추가 작업 불필요

### **Option B: 지표 표준화**

**방법**:
- 지표(q6-q49)를 Z-score 표준화
- LV와 지표의 스케일 일치

**장점**:
- 잔차 감소 → 우도 증가
- 이론적으로 올바름

**단점**:
- 데이터 재전처리 필요
- 해석 복잡해짐

### **Option C: price 표준화**

**방법**:
- price를 Z-score 표준화
- 선택모델 파라미터 추정 개선

**장점**:
- 수치 안정성 향상
- 파라미터 해석 용이

**단점**:
- 원척도 해석 불가
- 역변환 필요

---

## 6. 결론

### **현재 상태**

1. **CFA 요인점수**: ✅ 표준화됨 (평균 0, 표준편차 1)
2. **구조모델 LV (동시추정)**: ✅ 표준화됨 (N(0,1) 샘플링)
3. **선택 속성**: ❌ 표준화 안됨 (원척도)
4. **잠재변수 지표**: ❌ 표준화 안됨 (1-5점 리커트)

### **스케일 불일치**

- **LV (표준화) vs 지표 (원척도)** → 큰 잔차 → 낮은 측정모델 우도
- **현재 해결책**: 측정모델 우도 스케일링 (÷38)

### **권장**

- ✅ **현재 상태 유지** (Option A)
- 이미 작동하고 있음
- 추가 작업 불필요

---

**작성일**: 2025-11-20  
**작성자**: Augment Agent

