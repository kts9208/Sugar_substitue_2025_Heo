# 데이터 변수 Z-score 표준화 통합 계획

## 📋 개요

**목표:** 파라미터 스케일링은 유지하되, 데이터 변수(price, health_label 등)를 z-score 표준화하여 모델에 투입

**핵심 아이디어:**
- **파라미터 스케일링**: 최적화 안정성 (유지)
- **데이터 표준화**: 변수 스케일 균형 (추가)

---

## 1. 현재 상황 분석

### 1.1 기존 z-score 표준화 로직

**위치:** `scripts/standardize_price.py`

```python
def standardize_price(data: pd.DataFrame) -> pd.DataFrame:
    """가격 변수 Z-score 표준화"""
    df = data.copy()
    
    # 원본 통계
    price_values = df['price'].dropna()
    original_mean = price_values.mean()
    original_std = price_values.std(ddof=0)  # 모집단 표준편차
    
    # Z-score 표준화
    mask = df['price'].notna()
    df.loc[mask, 'price'] = (df.loc[mask, 'price'] - original_mean) / original_std
    
    return df
```

**특징:**
- ✅ 간단하고 명확한 로직
- ✅ 평균과 표준편차 저장 (역변환 가능)
- ⚠️ 단일 변수만 처리

### 1.2 데이터 변수 사용 위치

**선택모델 효용 계산:**
```python
# choice_equations.py, Line 112
X = data[self.choice_attributes].values  # ['health_label', 'price']

# 효용 계산
V = ASC + beta @ X + theta * LV
```

**현재 스케일:**
- `price`: 2000~6000 원 (매우 큼)
- `health_label`: 0~1 (적절)
- `LV`: -3~+3 (표준정규분포)

**문제:**
- `beta_price` 그래디언트: 2000~6000 (매우 큼)
- `beta_health_label` 그래디언트: 0.1~1 (적절)
- **6000배 차이** → 최적화 불안정

---

## 2. 통합 방안 설계

### 2.1 설계 원칙

1. **분리된 책임 (Separation of Concerns)**
   - 파라미터 스케일링: `ParameterScaler` (최적화 공간)
   - 데이터 표준화: 새로운 클래스 (데이터 공간)

2. **재사용성 (Reusability)**
   - 기존 z-score 로직 활용
   - 여러 변수 동시 처리

3. **투명성 (Transparency)**
   - 표준화 파라미터 저장 (평균, 표준편차)
   - 역변환 가능

4. **최소 침습성 (Minimal Invasiveness)**
   - 기존 코드 최소 수정
   - 기존 파라미터 스케일링 유지

### 2.2 제안 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Estimator                                │
│                                                             │
│  ┌──────────────────┐         ┌──────────────────┐        │
│  │ DataStandardizer │         │ ParameterScaler  │        │
│  │                  │         │                  │        │
│  │ - price          │         │ - gamma: 50.0    │        │
│  │ - health_label   │         │ - beta: 0.056    │        │
│  │ - ...            │         │ - ...            │        │
│  └──────────────────┘         └──────────────────┘        │
│         ↓                              ↓                   │
│  [표준화된 데이터]            [스케일링된 파라미터]        │
│         ↓                              ↓                   │
│  ┌──────────────────────────────────────────────┐         │
│  │         Likelihood & Gradient                │         │
│  └──────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 구현 방안

### 3.1 새로운 클래스: DataStandardizer

**파일:** `src/analysis/hybrid_choice_model/iclv_models/data_standardizer.py`

**기능:**
1. 여러 변수 동시 z-score 표준화
2. 표준화 파라미터 저장 (평균, 표준편차)
3. 역변환 지원
4. 로깅

**인터페이스:**
```python
class DataStandardizer:
    def __init__(self, variables_to_standardize: List[str], logger=None):
        """
        Args:
            variables_to_standardize: 표준화할 변수 리스트
                예: ['price', 'health_label']
        """
        
    def fit(self, data: pd.DataFrame) -> None:
        """데이터에서 평균과 표준편차 계산"""
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 표준화"""
        
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """fit + transform"""
        
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """역변환 (표준화 → 원본)"""
        
    def get_standardization_params(self) -> Dict[str, Dict[str, float]]:
        """표준화 파라미터 반환"""
```

### 3.2 통합 위치

**파일:** `simultaneous_estimator_fixed.py`

**초기화 단계 (estimate 메서드 시작 부분):**
```python
def estimate(self, data, measurement_model, structural_model, choice_model):
    # 1. 데이터 표준화 (새로 추가)
    if self.config.estimation.standardize_choice_attributes:
        self.data_standardizer = DataStandardizer(
            variables_to_standardize=config.choice.choice_attributes,
            logger=self.iteration_logger
        )
        data = self.data_standardizer.fit_transform(data)
        self.iteration_logger.info("선택 속성 z-score 표준화 완료")
    
    # 2. 파라미터 스케일링 (기존 유지)
    if use_parameter_scaling:
        custom_scales = self._get_custom_scales(param_names)
        self.param_scaler = ParameterScaler(...)
```

---

## 4. 상세 구현

### 4.1 DataStandardizer 클래스

**핵심 메서드:**

```python
def fit(self, data: pd.DataFrame) -> None:
    """평균과 표준편차 계산"""
    self.stats = {}
    
    for var in self.variables_to_standardize:
        if var not in data.columns:
            self.logger.warning(f"변수 '{var}'가 데이터에 없습니다. 건너뜁니다.")
            continue
        
        values = data[var].dropna()
        mean = values.mean()
        std = values.std(ddof=0)  # 모집단 표준편차
        
        self.stats[var] = {'mean': mean, 'std': std}
        
        self.logger.info(
            f"  {var:20s}: mean={mean:10.4f}, std={std:10.4f}"
        )

def transform(self, data: pd.DataFrame) -> pd.DataFrame:
    """z-score 표준화"""
    df = data.copy()
    
    for var, params in self.stats.items():
        if var not in df.columns:
            continue
        
        mean = params['mean']
        std = params['std']
        
        # NaN이 아닌 값만 표준화
        mask = df[var].notna()
        
        if std > 1e-10:  # 표준편차가 0이 아닌 경우만
            df.loc[mask, var] = (df.loc[mask, var] - mean) / std
        else:
            self.logger.warning(
                f"  {var}: 표준편차가 0에 가까워 중심화만 적용"
            )
            df.loc[mask, var] = df.loc[mask, var] - mean
    
    return df
```

### 4.2 Config 설정 추가

**파일:** `iclv_config.py`

```python
@dataclass
class EstimationConfig:
    # 기존 설정
    use_parameter_scaling: bool = True
    
    # 새로 추가
    standardize_choice_attributes: bool = True
    """선택 속성 z-score 표준화 여부"""
```

---

## 5. 사용 예시

### 5.1 표준화 활성화

```python
# config 설정
config.estimation.standardize_choice_attributes = True

# 자동으로 price, health_label 표준화
estimator.estimate(data, measurement_model, structural_model, choice_model)
```

**효과:**
```
Before:
  price: 2000~6000 원
  health_label: 0~1

After:
  price: -1.5~+1.5 (z-score)
  health_label: -1.0~+1.0 (z-score)
```

### 5.2 표준화 비활성화

```python
# config 설정
config.estimation.standardize_choice_attributes = False

# 원본 데이터 사용
estimator.estimate(data, measurement_model, structural_model, choice_model)
```

---

## 6. 파라미터 해석

### 6.1 표준화 전 (현재)

```python
beta_price = -0.56
```

**해석:** 가격이 1원 증가하면 효용이 -0.56 감소

### 6.2 표준화 후 (제안)

```python
beta_price_standardized = -560
```

**해석:** 가격이 1 표준편차 증가하면 효용이 -560 감소

**원본 스케일로 변환:**
```python
# price_std = 1000 (예시)
beta_price_original = beta_price_standardized / price_std
                    = -560 / 1000
                    = -0.56
```

**결론:** 효용함수 값은 동일, 해석도 동일

---

## 7. 장단점 분석

### 7.1 장점 ✅

1. **그래디언트 균형**
   - price 그래디언트: 2000~6000 → 2~6 (1000배 감소)
   - health_label 그래디언트: 0.1~1 → 0.1~1 (동일)
   - **균형잡힌 그래디언트** → 최적화 안정성 향상

2. **파라미터 스케일링과 독립적**
   - 데이터 표준화: 데이터 공간
   - 파라미터 스케일링: 최적화 공간
   - **두 메커니즘이 독립적으로 작동**

3. **재사용성**
   - 다른 모델에서도 사용 가능
   - 순차추정, 동시추정 모두 적용 가능

4. **투명성**
   - 표준화 파라미터 저장
   - 역변환 가능
   - 해석 가능

### 7.2 단점 ⚠️

1. **복잡성 증가**
   - 새로운 클래스 추가
   - 데이터 전처리 단계 추가

2. **파라미터 해석 주의**
   - 표준화된 스케일로 추정
   - 원본 스케일로 변환 필요

3. **초기값 조정 필요**
   - 표준화 후 beta 초기값 조정 필요
   - 예: `beta_price = 0.1` → `beta_price = -500`

---

## 8. 구현 우선순위

### Phase 1: DataStandardizer 클래스 생성
- [ ] `data_standardizer.py` 파일 생성
- [ ] `DataStandardizer` 클래스 구현
- [ ] 단위 테스트 작성

### Phase 2: Config 설정 추가
- [ ] `EstimationConfig`에 `standardize_choice_attributes` 추가
- [ ] 기본값 설정 (`True` 권장)

### Phase 3: Estimator 통합
- [ ] `simultaneous_estimator_fixed.py`에 통합
- [ ] 초기화 단계에서 데이터 표준화
- [ ] 로깅 추가

### Phase 4: 테스트 및 검증
- [ ] 표준화 전후 비교
- [ ] 그래디언트 크기 비교
- [ ] 최적화 수렴 속도 비교

---

## 9. 예상 효과

### 9.1 그래디언트 크기 비교

**Before (표준화 전):**
```
grad(beta_price): 2000~6000
grad(beta_health_label): 0.1~1
비율: 6000:1
```

**After (표준화 후):**
```
grad(beta_price): 2~6
grad(beta_health_label): 0.1~1
비율: 6:1 (1000배 개선)
```

### 9.2 최적화 안정성

**Before:**
- L-BFGS-B가 beta_price에 집중
- 다른 파라미터 업데이트 느림
- 수렴 느림

**After:**
- 모든 파라미터 균형있게 업데이트
- 수렴 빠름
- 안정적

---

## 10. 결론

**권장 사항:**
1. ✅ **DataStandardizer 클래스 생성** (새로운 파일)
2. ✅ **파라미터 스케일링 유지** (기존 로직)
3. ✅ **두 메커니즘 독립적 운영**

**핵심 공식:**
```
# 데이터 표준화
X_standardized = (X - mean(X)) / std(X)

# 파라미터 스케일링 (기존 유지)
θ_internal = θ_external / scale
```

**예상 효과:**
- 그래디언트 균형: 1000배 개선
- 최적화 안정성: 향상
- 수렴 속도: 향상

**다음 단계:**
- DataStandardizer 클래스 구현
- 통합 테스트
- 성능 비교

