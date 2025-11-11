# 구조모델 수정 가능성 검토 보고서

**작성일**: 2025-11-11  
**검토 대상**: ICLV 구조모델 계층적 구조 및 조절효과 추가

---

## 📋 요청사항 요약

### 제안된 수정사항

1. **계층적 구조 추가**: 건강관심도 → 건강유익성 → 구매의도
2. **조절효과 추가**: 인지된 가격수준과 영양지식이 구매의도 → 선택 관계를 조절
3. **사회인구학적 변수 효과 제거**: 구매의도에 대한 직접 효과 제거
4. **기존 gamma_lv 경로 제거**: 4개 외생 LV → 구매의도 경로 제거

---

## 🎯 목표 구조

### **현재 구조**
```
외생 LV (독립):
  - 건강관심도 = η₁ ~ N(0,1)
  - 건강유익성 = η₂ ~ N(0,1)
  - 가격수준 = η₃ ~ N(0,1)
  - 영양지식 = η₄ ~ N(0,1)

내생 LV:
  구매의도 = γ₁·건강관심도 + γ₂·건강유익성 + γ₃·가격수준 + γ₄·영양지식
           + γ₅·age + γ₆·gender + γ₇·income + η

선택모델:
  V = intercept + β·X + λ·구매의도
  P(choice=1) = Φ(V)
```

### **제안된 구조**
```
1차 LV (외생):
  - 건강관심도 = η₁ ~ N(0,1)
  - 가격수준 = η₂ ~ N(0,1)
  - 영양지식 = η₃ ~ N(0,1)

2차 LV (중간 내생):
  건강유익성 = γ₁·건강관심도 + η₂

3차 LV (최종 내생):
  구매의도 = γ₂·건강유익성 + η₃

선택모델 (조절효과 포함):
  V = intercept + β·X + λ₁·구매의도 + λ₂·(구매의도 × 가격수준) + λ₃·(구매의도 × 영양지식)
  P(choice=1) = Φ(V)
```

---

## ✅ 가능성 분석

### **1. 계층적 구조 (건강관심도 → 건강유익성 → 구매의도)**

#### **결론**: ✅ **가능 (중간 난이도)**

#### **필요한 수정사항**:

##### **1.1 MultiLatentStructuralConfig 확장**
```python
@dataclass
class HierarchicalStructuralConfig:
    """
    계층적 구조모델 설정
    
    구조:
    - 1차 LV (외생): LV_i = η_i ~ N(0, 1)
    - 2차 LV (중간): LV_mid = Σ(γ_i * LV_1st) + η
    - 3차 LV (최종): LV_final = Σ(γ_j * LV_2nd) + η
    """
    
    # 계층 정의
    first_order_lvs: List[str]  # ['health_concern', 'perceived_price', 'nutrition_knowledge']
    second_order_lvs: Dict[str, List[str]]  # {'perceived_benefit': ['health_concern']}
    third_order_lv: str  # 'purchase_intention'
    
    # 경로 정의
    paths: Dict[str, List[str]]  # {'perceived_benefit': ['health_concern'],
                                  #  'purchase_intention': ['perceived_benefit']}
    
    # 오차 분산
    error_variances: Dict[str, float]  # {'perceived_benefit': 1.0, 'purchase_intention': 1.0}
```

##### **1.2 MultiLatentStructural 클래스 수정**
```python
class HierarchicalStructural:
    """계층적 구조모델"""
    
    def predict(self, data: pd.DataFrame,
                first_order_draws: np.ndarray,
                params: Dict[str, np.ndarray],
                second_order_draws: Dict[str, float],
                third_order_draw: float) -> Dict[str, float]:
        """
        계층적 잠재변수 예측
        
        순서:
        1. 1차 LV (외생) = draws
        2. 2차 LV (중간) = f(1차 LV) + error
        3. 3차 LV (최종) = f(2차 LV) + error
        """
        latent_vars = {}
        
        # 1차 LV (외생)
        for i, lv_name in enumerate(self.first_order_lvs):
            latent_vars[lv_name] = first_order_draws[i]
        
        # 2차 LV (중간) - 건강유익성
        # perceived_benefit = γ₁ * health_concern + η
        gamma_hc_to_pb = params['gamma_health_concern_to_perceived_benefit']
        latent_vars['perceived_benefit'] = (
            gamma_hc_to_pb * latent_vars['health_concern'] +
            np.sqrt(self.error_variances['perceived_benefit']) * second_order_draws['perceived_benefit']
        )
        
        # 3차 LV (최종) - 구매의도
        # purchase_intention = γ₂ * perceived_benefit + η
        gamma_pb_to_pi = params['gamma_perceived_benefit_to_purchase_intention']
        latent_vars['purchase_intention'] = (
            gamma_pb_to_pi * latent_vars['perceived_benefit'] +
            np.sqrt(self.error_variances['purchase_intention']) * third_order_draw
        )
        
        return latent_vars
```

##### **1.3 로그우도 함수 수정**
```python
def log_likelihood(self, data: pd.DataFrame,
                  latent_vars: Dict[str, float],
                  first_order_draws: np.ndarray,
                  params: Dict[str, np.ndarray],
                  second_order_draws: Dict[str, float],
                  third_order_draw: float) -> float:
    """
    계층적 구조모델 로그우도
    
    LL = Σ log P(LV_1st) + Σ log P(LV_2nd | LV_1st) + log P(LV_3rd | LV_2nd)
    """
    ll = 0.0
    
    # 1차 LV: N(0, 1)
    for lv_name in self.first_order_lvs:
        ll += norm.logpdf(latent_vars[lv_name], loc=0, scale=1)
    
    # 2차 LV: N(γ₁ * LV_1st, σ²)
    gamma_hc_to_pb = params['gamma_health_concern_to_perceived_benefit']
    pb_mean = gamma_hc_to_pb * latent_vars['health_concern']
    ll += norm.logpdf(
        latent_vars['perceived_benefit'],
        loc=pb_mean,
        scale=np.sqrt(self.error_variances['perceived_benefit'])
    )
    
    # 3차 LV: N(γ₂ * LV_2nd, σ²)
    gamma_pb_to_pi = params['gamma_perceived_benefit_to_purchase_intention']
    pi_mean = gamma_pb_to_pi * latent_vars['perceived_benefit']
    ll += norm.logpdf(
        latent_vars['purchase_intention'],
        loc=pi_mean,
        scale=np.sqrt(self.error_variances['purchase_intention'])
    )
    
    return ll
```

#### **파라미터 변화**:
- **현재**: gamma_lv (4개) + gamma_x (3개) = 7개
- **수정 후**: gamma_hc_to_pb (1개) + gamma_pb_to_pi (1개) = **2개**

#### **장점**:
- ✅ 이론적으로 타당한 계층 구조
- ✅ 파라미터 수 감소 (7개 → 2개)
- ✅ 해석 용이성 증가
- ✅ 간접효과 분석 가능 (건강관심도 → 건강유익성 → 구매의도)

#### **단점**:
- ⚠️ 가격수준, 영양지식이 구매의도에 직접 영향 없음 (이론적 타당성 검토 필요)
- ⚠️ 코드 수정 범위 중간 (구조모델 전체 재작성)

---

### **2. 조절효과 추가 (가격수준, 영양지식이 구매의도 → 선택 관계 조절)**

#### **결론**: ✅ **가능 (쉬움)**

#### **필요한 수정사항**:

##### **2.1 BinaryProbitChoice 클래스 수정**
```python
class BinaryProbitChoiceWithModeration:
    """
    조절효과가 포함된 Binary Probit 선택모델
    
    Model:
        V = intercept + β·X + λ₁·LV_main + λ₂·(LV_main × LV_mod1) + λ₃·(LV_main × LV_mod2)
        P(Yes) = Φ(V)
    
    여기서:
        - LV_main: 주 잠재변수 (구매의도)
        - LV_mod1: 조절변수 1 (가격수준)
        - LV_mod2: 조절변수 2 (영양지식)
    """
    
    def log_likelihood(self, data: pd.DataFrame,
                      latent_vars: Dict[str, float],
                      params: Dict) -> float:
        """
        조절효과 포함 로그우도
        
        V = intercept + β·X + λ₁·PI + λ₂·(PI × PP) + λ₃·(PI × NK)
        """
        intercept = params['intercept']
        beta = params['beta']
        lambda_main = params['lambda_main']  # 구매의도 주효과
        lambda_mod_price = params['lambda_mod_price']  # 가격수준 조절효과
        lambda_mod_knowledge = params['lambda_mod_knowledge']  # 영양지식 조절효과
        
        # 선택 속성
        X = data[self.choice_attributes].values
        choice = data['choice'].values
        
        # 잠재변수
        lv_main = latent_vars['purchase_intention']
        lv_mod_price = latent_vars['perceived_price']
        lv_mod_knowledge = latent_vars['nutrition_knowledge']
        
        # 효용 계산 (조절효과 포함)
        V = (intercept + X @ beta +
             lambda_main * lv_main +
             lambda_mod_price * (lv_main * lv_mod_price) +
             lambda_mod_knowledge * (lv_main * lv_mod_knowledge))
        
        # 확률 및 로그우도
        prob_yes = norm.cdf(V)
        prob_yes = np.clip(prob_yes, 1e-10, 1 - 1e-10)
        
        ll = np.sum(
            choice * np.log(prob_yes) +
            (1 - choice) * np.log(1 - prob_yes)
        )
        
        return ll
```

##### **2.2 ChoiceConfig 확장**
```python
@dataclass
class ChoiceConfig:
    """선택모델 설정"""
    
    choice_attributes: List[str]
    choice_type: str = 'binary'
    price_variable: str = 'price'
    
    # 조절효과 설정 (NEW)
    moderation_enabled: bool = False
    moderator_lvs: Optional[List[str]] = None  # ['perceived_price', 'nutrition_knowledge']
    main_lv: str = 'purchase_intention'
```

#### **파라미터 변화**:
- **현재**: intercept (1) + beta (3) + lambda (1) = 5개
- **수정 후**: intercept (1) + beta (3) + lambda_main (1) + lambda_mod_price (1) + lambda_mod_knowledge (1) = **7개**

#### **장점**:
- ✅ 구현 난이도 낮음 (선택모델만 수정)
- ✅ 이론적으로 타당 (조절효과는 선택모델에서 흔히 사용)
- ✅ 해석 용이 (단순 곱셈 상호작용)
- ✅ 기존 조절효과 분석 코드 활용 가능 (`src/analysis/moderation_analysis/`)

#### **단점**:
- ⚠️ 파라미터 수 증가 (5개 → 7개)
- ⚠️ 다중공선성 가능성 (상호작용항)

---

### **3. 사회인구학적 변수 효과 제거**

#### **결론**: ✅ **가능 (매우 쉬움)**

#### **필요한 수정사항**:
```python
# 현재
structural_config = MultiLatentStructuralConfig(
    endogenous_lv='purchase_intention',
    exogenous_lvs=['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge'],
    covariates=['age_std', 'gender', 'income_std'],  # ← 제거
    error_variance=1.0
)

# 수정 후
structural_config = HierarchicalStructuralConfig(
    first_order_lvs=['health_concern', 'perceived_price', 'nutrition_knowledge'],
    paths={
        'perceived_benefit': ['health_concern'],
        'purchase_intention': ['perceived_benefit']
    },
    covariates=[],  # ← 빈 리스트
    error_variances={'perceived_benefit': 1.0, 'purchase_intention': 1.0}
)
```

#### **파라미터 변화**:
- gamma_x (3개) 제거

#### **장점**:
- ✅ 구현 매우 쉬움
- ✅ 파라미터 수 감소
- ✅ 모델 단순화

#### **단점**:
- ⚠️ 사회인구학적 변수의 효과를 완전히 무시 (이론적 타당성 검토 필요)
- ⚠️ 대안: 사회인구학적 변수 → 1차 LV (건강관심도) 경로 추가 고려

---

### **4. 기존 gamma_lv 경로 제거**

#### **결론**: ✅ **가능 (계층적 구조에 포함됨)**

계층적 구조를 구현하면 자동으로 해결됩니다.

- **현재**: 4개 외생 LV → 구매의도 (직접 경로)
- **수정 후**: 건강관심도 → 건강유익성 → 구매의도 (간접 경로만)

---

## 📊 전체 파라미터 비교

| 모델 구성요소 | 현재 | 수정 후 | 변화 |
|--------------|------|---------|------|
| **측정모델** | | | |
| - 건강관심도 (6 indicators) | ζ(6) + τ(6×4) | 동일 | - |
| - 건강유익성 (6 indicators) | ζ(6) + τ(6×4) | 동일 | - |
| - 가격수준 (3 indicators) | ζ(3) + τ(3×4) | 동일 | - |
| - 영양지식 (20 indicators) | ζ(20) + τ(20×4) | 동일 | - |
| - 구매의도 (3 indicators) | ζ(3) + τ(3×4) | 동일 | - |
| **구조모델** | | | |
| - gamma_lv | 4개 | 0개 | -4 |
| - gamma_x | 3개 | 0개 | -3 |
| - gamma (계층 경로) | 0개 | 2개 | +2 |
| **선택모델** | | | |
| - intercept | 1개 | 1개 | - |
| - beta | 3개 | 3개 | - |
| - lambda | 1개 | 0개 | -1 |
| - lambda_main | 0개 | 1개 | +1 |
| - lambda_mod | 0개 | 2개 | +2 |
| **총 구조+선택** | 12개 | 9개 | **-3개** |

---

## 🔧 구현 난이도 평가

| 수정사항 | 난이도 | 예상 소요 시간 | 주요 작업 |
|---------|--------|---------------|----------|
| 1. 계층적 구조 | ⭐⭐⭐ 중간 | 4-6시간 | `HierarchicalStructural` 클래스 신규 작성 |
| 2. 조절효과 | ⭐ 쉬움 | 2-3시간 | `BinaryProbitChoice` 수정 |
| 3. 사회인구학적 변수 제거 | ⭐ 매우 쉬움 | 10분 | Config 수정 |
| 4. gamma_lv 경로 제거 | - | - | 1번에 포함 |
| **전체** | ⭐⭐ 중하 | **6-9시간** | |

---

## ⚠️ 주의사항 및 권장사항

### **1. 이론적 타당성 검토 필요**

#### **질문 1**: 가격수준과 영양지식이 구매의도에 직접 영향을 주지 않는가?
- 현재 제안: 가격수준, 영양지식 → 구매의도 경로 없음
- 대안: 가격수준, 영양지식 → 구매의도 경로 추가 고려

#### **질문 2**: 사회인구학적 변수의 역할은?
- 현재 제안: 완전 제거
- 대안 1: 사회인구학적 변수 → 건강관심도 (MIMIC 모델)
- 대안 2: 사회인구학적 변수 → 선택모델 직접 효과

### **2. 모델 식별 (Identification) 검토**

계층적 구조에서 모든 잠재변수의 분산이 식별 가능한지 확인 필요:
- 1차 LV: N(0, 1) 고정 ✅
- 2차 LV: 오차 분산 고정 필요 ✅
- 3차 LV: 오차 분산 고정 필요 ✅

### **3. 테스트 전략**

1. **단계별 구현**:
   - Step 1: 계층적 구조만 구현 (조절효과 없이)
   - Step 2: 조절효과 추가
   - Step 3: 통합 테스트

2. **검증**:
   - 시뮬레이션 데이터로 파라미터 복원 테스트
   - 실제 데이터로 수렴성 확인
   - 기존 모델과 적합도 비교 (AIC, BIC)

---

## 📝 최종 권장사항

### ✅ **권장: 단계적 구현**

#### **Phase 1: 계층적 구조 (우선 구현)**
```
건강관심도 → 건강유익성 → 구매의도 → 선택
```
- 이론적으로 타당
- 파라미터 감소
- 해석 용이

#### **Phase 2: 조절효과 추가**
```
V = β·X + λ₁·PI + λ₂·(PI × PP) + λ₃·(PI × NK)
```
- 구현 쉬움
- 이론적 기여도 높음

#### **Phase 3: 사회인구학적 변수 재검토**
- 완전 제거 vs. 1차 LV에 효과 추가 vs. 선택모델 직접 효과
- 이론 및 데이터 기반 결정

---

## 🎯 결론

**모든 제안사항은 기술적으로 구현 가능합니다.**

- ✅ 계층적 구조: 가능 (중간 난이도)
- ✅ 조절효과: 가능 (쉬움)
- ✅ 사회인구학적 변수 제거: 가능 (매우 쉬움)
- ✅ gamma_lv 경로 제거: 가능 (계층적 구조에 포함)

**예상 총 작업 시간**: 6-9시간

**다음 단계**: 이론적 타당성 확인 후 구현 시작 여부 결정

