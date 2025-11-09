# 다중 잠재변수 ICLV 구현 계획

**날짜**: 2025-11-09  
**목적**: 5개 잠재변수 완전 동시추정 구현 (기존 코드 최대 재사용)

---

## 🎯 목표 구조

### **5개 잠재변수**

1. **건강관심도** (LV1): Q6-Q11 (6개 문항)
2. **건강유익성** (LV2): Q12-Q17 (6개 문항)
3. **가격수준** (LV3): Q27-Q29 (3개 문항)
4. **영양지식** (LV4): Q30-Q49 (20개 문항)
5. **구매의도** (LV5): Q18-Q20 (3개 문항) - **내생변수**

### **모델 구조**

```
[측정모델]
LV1 =~ ζ1*q6 + ζ2*q7 + ... + ζ6*q11
LV2 =~ ζ7*q12 + ζ8*q13 + ... + ζ12*q17
LV3 =~ ζ13*q27 + ζ14*q28 + ζ15*q29
LV4 =~ ζ16*q30 + ζ17*q31 + ... + ζ35*q49
LV5 =~ ζ36*q18 + ζ37*q19 + ζ38*q20

[구조모델]
LV1 = η1 (외생)
LV2 = η2 (외생)
LV3 = η3 (외생)
LV4 = η4 (외생)
LV5 = γ1*LV1 + γ2*LV2 + γ3*LV3 + γ4*LV4 
    + γ5*age + γ6*gender + γ7*income + γ8*education + η5

[선택모델]
V = β0 + β1*sugar_free + β2*health_label + β3*price + λ*LV5
P(선택) = Φ(V)
```

---

## 🔧 구현 전략

### **핵심 아이디어**: 기존 클래스를 **컨테이너**로 감싸기

기존 코드:
- `OrderedProbitMeasurement`: 1개 LV 측정모델
- `LatentVariableRegression`: 1개 LV 구조모델
- `BinaryProbitChoice`: 선택모델
- `SimultaneousEstimator`: 동시추정 엔진

새로운 코드:
- `MultiLatentMeasurement`: 5개 측정모델 컨테이너
- `MultiLatentStructural`: 5개 LV 구조모델 (4개 외생 + 1개 내생)
- `MultiLatentSimultaneousEstimator`: 다중 LV 동시추정 엔진

---

## 📁 파일 구조

```
src/analysis/hybrid_choice_model/iclv_models/
├─ measurement_equations.py          (기존 - 수정 없음)
├─ structural_equations.py           (기존 - 수정 없음)
├─ choice_equations.py               (기존 - 수정 없음)
├─ simultaneous_estimator_fixed.py   (기존 - 수정 없음)
├─ multi_latent_measurement.py       (신규 - 측정모델 컨테이너)
├─ multi_latent_structural.py        (신규 - 구조모델 확장)
├─ multi_latent_config.py            (신규 - 설정)
└─ multi_latent_estimator.py         (신규 - 동시추정 엔진)
```

---

## 📋 구현 단계

### **Phase 1: 설정 클래스 (30분)**

**파일**: `multi_latent_config.py`

```python
@dataclass
class MultiLatentConfig:
    """다중 잠재변수 ICLV 설정"""
    
    # 측정모델 설정 (5개)
    measurement_configs: Dict[str, MeasurementConfig]
    
    # 구조모델 설정
    endogenous_lv: str  # 'purchase_intention'
    exogenous_lvs: List[str]  # ['health_concern', 'perceived_benefit', ...]
    covariates: List[str]  # ['age_std', 'gender', ...]
    
    # 선택모델 설정
    choice_config: ChoiceConfig
    
    # 추정 설정
    estimation_config: EstimationConfig
    
    # 데이터 설정
    individual_id_column: str = 'respondent_id'
```

---

### **Phase 2: 측정모델 컨테이너 (1시간)**

**파일**: `multi_latent_measurement.py`

**핵심**: 기존 `OrderedProbitMeasurement`를 5개 생성하여 관리

```python
class MultiLatentMeasurement:
    """
    다중 잠재변수 측정모델 컨테이너
    
    기존 OrderedProbitMeasurement를 재사용
    """
    
    def __init__(self, configs: Dict[str, MeasurementConfig]):
        """
        Args:
            configs: {
                'health_concern': MeasurementConfig(...),
                'perceived_benefit': MeasurementConfig(...),
                ...
            }
        """
        self.configs = configs
        self.models = {}
        
        # 각 LV에 대해 기존 측정모델 생성
        for lv_name, config in configs.items():
            self.models[lv_name] = OrderedProbitMeasurement(config)
    
    def log_likelihood(self, data: pd.DataFrame, 
                      latent_vars: Dict[str, float],
                      params: Dict) -> float:
        """
        전체 측정모델 로그우도
        
        LL = Σ_j LL_j(Indicators_j | LV_j)
        
        Args:
            latent_vars: {'health_concern': 0.5, 'perceived_benefit': 0.3, ...}
            params: {
                'health_concern': {'zeta': ..., 'tau': ...},
                'perceived_benefit': {'zeta': ..., 'tau': ...},
                ...
            }
        """
        total_ll = 0.0
        
        for lv_name, model in self.models.items():
            lv = latent_vars[lv_name]
            lv_params = params[lv_name]
            
            # 기존 측정모델 재사용
            ll = model.log_likelihood(data, lv, lv_params)
            total_ll += ll
        
        return total_ll
    
    def get_n_parameters(self) -> int:
        """총 파라미터 수"""
        total = 0
        for model in self.models.values():
            total += model.get_n_parameters()
        return total
```

---

### **Phase 3: 구조모델 확장 (1.5시간)**

**파일**: `multi_latent_structural.py`

**핵심**: 4개 외생 LV + 1개 내생 LV 구조

```python
class MultiLatentStructural:
    """
    다중 잠재변수 구조모델
    
    외생 LV: LV_i = η_i ~ N(0, 1)
    내생 LV: LV_endo = Σ(γ_i * LV_i) + Σ(γ_j * X_j) + η
    """
    
    def __init__(self, endogenous_lv: str, exogenous_lvs: List[str],
                 covariates: List[str], error_variance: float = 1.0):
        self.endogenous_lv = endogenous_lv
        self.exogenous_lvs = exogenous_lvs
        self.covariates = covariates
        self.error_variance = error_variance
        
        self.n_exo = len(exogenous_lvs)
        self.n_cov = len(covariates)
    
    def predict(self, data: pd.DataFrame, 
                exo_draws: np.ndarray,
                params: Dict,
                endo_draw: float) -> Dict[str, float]:
        """
        모든 잠재변수 예측
        
        Args:
            exo_draws: 외생 LV draws (n_exo,)
            params: {
                'gamma_lv': np.ndarray (n_exo,),
                'gamma_x': np.ndarray (n_cov,)
            }
            endo_draw: 내생 LV 오차항 draw
        
        Returns:
            {'health_concern': 0.5, 'perceived_benefit': 0.3, ..., 'purchase_intention': 0.8}
        """
        latent_vars = {}
        
        # 1. 외생 LV (표준정규분포)
        for i, lv_name in enumerate(self.exogenous_lvs):
            latent_vars[lv_name] = exo_draws[i]
        
        # 2. 내생 LV
        gamma_lv = params['gamma_lv']
        gamma_x = params['gamma_x']
        
        # 외생 LV 효과
        lv_effect = np.sum(gamma_lv * exo_draws)
        
        # 공변량 효과
        X = data[self.covariates].values.flatten()
        x_effect = np.sum(gamma_x * X)
        
        # 내생 LV
        latent_vars[self.endogenous_lv] = lv_effect + x_effect + np.sqrt(self.error_variance) * endo_draw
        
        return latent_vars
    
    def log_likelihood(self, latent_vars: Dict[str, float],
                      exo_draws: np.ndarray,
                      params: Dict,
                      endo_draw: float) -> float:
        """
        구조모델 로그우도
        
        LL = Σ log P(LV_exo) + log P(LV_endo | LV_exo, X)
        """
        ll = 0.0
        
        # 외생 LV: N(0, 1)
        for i, lv_name in enumerate(self.exogenous_lvs):
            lv = latent_vars[lv_name]
            ll += norm.logpdf(lv, loc=0, scale=1)
        
        # 내생 LV: N(γ*LV + γ*X, σ²)
        gamma_lv = params['gamma_lv']
        gamma_x = params['gamma_x']
        
        lv_effect = np.sum(gamma_lv * exo_draws)
        # X는 predict에서 이미 계산됨
        
        lv_endo = latent_vars[self.endogenous_lv]
        lv_endo_mean = lv_effect + np.sum(gamma_x * X)  # X 필요
        
        ll += norm.logpdf(lv_endo, loc=lv_endo_mean, scale=np.sqrt(self.error_variance))
        
        return ll
    
    def get_n_parameters(self) -> int:
        """파라미터 수: n_exo + n_cov"""
        return self.n_exo + self.n_cov
```

---

### **Phase 4: 동시추정 엔진 (2-3시간)**

**파일**: `multi_latent_estimator.py`

**핵심**: 기존 `SimultaneousEstimator` 로직 재사용

```python
class MultiLatentSimultaneousEstimator:
    """
    다중 잠재변수 동시추정 엔진
    
    기존 SimultaneousEstimator 로직을 확장
    """
    
    def _compute_individual_likelihood(self, ind_data, ind_draws, param_dict):
        """
        개인별 우도 계산
        
        ind_draws: (n_draws, n_exo + 1)
            - [:, :n_exo]: 외생 LV draws
            - [:, n_exo]: 내생 LV draw
        """
        draw_lls = []
        
        for draw_idx in range(len(ind_draws)):
            # 1. Draws 분리
            exo_draws = ind_draws[draw_idx, :self.n_exo]
            endo_draw = ind_draws[draw_idx, self.n_exo]
            
            # 2. 구조모델: 모든 LV 예측
            latent_vars = self.structural_model.predict(
                ind_data, exo_draws, param_dict['structural'], endo_draw
            )
            
            # 3. 측정모델 우도 (5개 LV)
            ll_measurement = self.measurement_model.log_likelihood(
                ind_data, latent_vars, param_dict['measurement']
            )
            
            # 4. 선택모델 우도 (내생 LV만 사용)
            lv_endo = latent_vars[self.endogenous_lv]
            ll_choice = 0.0
            for idx in range(len(ind_data)):
                ll_choice += self.choice_model.log_likelihood(
                    ind_data.iloc[idx:idx+1],
                    lv_endo,
                    param_dict['choice']
                )
            
            # 5. 구조모델 우도
            ll_structural = self.structural_model.log_likelihood(
                latent_vars, exo_draws, param_dict['structural'], endo_draw
            )
            
            # 6. 결합 로그우도
            draw_ll = ll_measurement + ll_choice + ll_structural
            draw_lls.append(draw_ll)
        
        # logsumexp
        person_ll = logsumexp(draw_lls) - np.log(len(draw_lls))
        return person_ll
```

---

## 📊 파라미터 구조

### **총 파라미터 수**

| 모델 | 파라미터 | 개수 |
|------|---------|------|
| **측정모델** | | |
| - 건강관심도 | ζ (6) + τ (24) | 30 |
| - 건강유익성 | ζ (6) + τ (24) | 30 |
| - 가격수준 | ζ (3) + τ (12) | 15 |
| - 영양지식 | ζ (20) + τ (80) | 100 |
| - 구매의도 | ζ (3) + τ (12) | 15 |
| **구조모델** | γ_lv (4) + γ_x (4) | 8 |
| **선택모델** | β (4) + λ (1) | 5 |
| **총계** | | **203개** |

---

## ⏱️ 예상 소요 시간

| Phase | 작업 | 시간 |
|-------|------|------|
| 1 | 설정 클래스 | 30분 |
| 2 | 측정모델 컨테이너 | 1시간 |
| 3 | 구조모델 확장 | 1.5시간 |
| 4 | 동시추정 엔진 | 2-3시간 |
| 5 | 테스트 스크립트 | 1시간 |
| 6 | 디버깅 | 2-3시간 |
| **총계** | | **8-10시간** |

---

## 🎯 다음 단계

1. Phase 1부터 순차적으로 구현
2. 각 Phase마다 단위 테스트
3. 전체 통합 테스트
4. 실제 데이터로 추정

시작하시겠습니까?

