# 조절효과 구현 상세 가이드

**작성일**: 2025-11-11  
**목적**: ICLV 선택모델에서 조절효과를 구현하는 방법 상세 설명

---

## 📊 조절효과란?

### 정의
**조절효과 (Moderation Effect)**: 독립변수(X)와 종속변수(Y) 간 관계의 강도나 방향이 제3의 변수(M, 조절변수)에 따라 달라지는 현상

### 수식
```
Y = β₀ + β₁·X + β₂·M + β₃·(X × M) + ε

여기서:
- β₁: X의 주효과 (Main Effect)
- β₂: M의 주효과
- β₃: 조절효과 (Moderation Effect) ← 핵심!
```

**해석**:
- β₃ > 0: M이 클수록 X→Y 관계가 강해짐 (정적 조절)
- β₃ < 0: M이 클수록 X→Y 관계가 약해짐 (부적 조절)
- β₃ = 0: 조절효과 없음

---

## 🎯 ICLV에서의 조절효과

### 현재 모델 (조절효과 없음)
```
V = intercept + β·X + λ·LV_main
P(choice=1) = Φ(V)

여기서:
- X: 선택 속성 (sugar_free, health_label, price)
- LV_main: 구매의도 (purchase_intention)
- λ: 구매의도의 선택에 대한 효과 (고정)
```

**문제점**: λ가 모든 사람에게 동일 → 현실적이지 않음

---

### 제안 모델 (조절효과 포함)
```
V = intercept + β·X + λ₁·PI + λ₂·(PI × PP) + λ₃·(PI × NK)
P(choice=1) = Φ(V)

여기서:
- PI: 구매의도 (Purchase Intention)
- PP: 가격수준 (Perceived Price)
- NK: 영양지식 (Nutrition Knowledge)
- λ₁: 구매의도 주효과
- λ₂: 가격수준 조절효과 (예상: λ₂ < 0)
- λ₃: 영양지식 조절효과 (예상: λ₃ > 0)
```

**의미**:
- 구매의도가 선택에 미치는 영향이 가격수준과 영양지식에 따라 달라짐
- 가격수준이 높으면 구매의도가 있어도 선택 확률 감소
- 영양지식이 높으면 구매의도가 선택으로 전환될 확률 증가

---

## 💻 구현 방법

### Step 1: 현재 코드 구조 이해

**현재 `BinaryProbitChoice.log_likelihood()` 핵심 부분**:

```python
# 효용 계산 (현재)
V = intercept + X @ beta + lambda_lv * lv_array

# 확률 계산
prob_yes = norm.cdf(V)

# 로그우도
ll = np.sum(choice * np.log(prob_yes) + (1 - choice) * np.log(1 - prob_yes))
```

**입력**:
- `lv`: 단일 잠재변수 (구매의도만)
- `params['lambda']`: 단일 계수

---

### Step 2: 조절효과 포함 수정

#### **2.1 입력 변경**

**Before**:
```python
def log_likelihood(self, data: pd.DataFrame, lv: np.ndarray, params: Dict) -> float:
    # lv: 구매의도만 (스칼라 또는 1D 배열)
```

**After**:
```python
def log_likelihood(self, data: pd.DataFrame, 
                  latent_vars: Dict[str, float],  # ← 변경!
                  params: Dict) -> float:
    """
    Args:
        latent_vars: 모든 잠재변수 값
            {
                'purchase_intention': 0.5,
                'perceived_price': -0.3,
                'nutrition_knowledge': 0.8
            }
    """
```

#### **2.2 파라미터 변경**

**Before**:
```python
params = {
    'intercept': 0.0,
    'beta': np.array([-2.0, 0.3, -1.5]),  # [sugar_free, health_label, price]
    'lambda': 1.0  # 구매의도 계수
}
```

**After**:
```python
params = {
    'intercept': 0.0,
    'beta': np.array([-2.0, 0.3, -1.5]),
    'lambda_main': 1.0,        # 구매의도 주효과
    'lambda_mod_price': -0.3,  # 가격수준 조절효과 (부적)
    'lambda_mod_knowledge': 0.2  # 영양지식 조절효과 (정적)
}
```

#### **2.3 효용 함수 수정**

**Before**:
```python
# 효용 계산 (조절효과 없음)
V = intercept + X @ beta + lambda_lv * lv_array
```

**After**:
```python
# 잠재변수 추출
lv_main = latent_vars['purchase_intention']
lv_mod_price = latent_vars['perceived_price']
lv_mod_knowledge = latent_vars['nutrition_knowledge']

# 효용 계산 (조절효과 포함)
V = (intercept + 
     X @ beta + 
     lambda_main * lv_main +                              # 주효과
     lambda_mod_price * (lv_main * lv_mod_price) +        # 조절효과 1
     lambda_mod_knowledge * (lv_main * lv_mod_knowledge)) # 조절효과 2
```

---

### Step 3: 완전한 구현 코드

```python
class BinaryProbitChoiceWithModeration:
    """조절효과가 포함된 Binary Probit 선택모델"""
    
    def __init__(self, config: ChoiceConfig,
                 main_lv: str = 'purchase_intention',
                 moderator_lvs: List[str] = None):
        """
        Args:
            config: 선택모델 설정
            main_lv: 주 잠재변수 (구매의도)
            moderator_lvs: 조절변수 리스트 ['perceived_price', 'nutrition_knowledge']
        """
        self.config = config
        self.choice_attributes = config.choice_attributes
        self.main_lv = main_lv
        self.moderator_lvs = moderator_lvs or []
        
        self.n_attributes = len(self.choice_attributes)
        self.n_moderators = len(self.moderator_lvs)
    
    def log_likelihood(self, data: pd.DataFrame,
                      latent_vars: Dict[str, float],
                      params: Dict) -> float:
        """
        조절효과 포함 로그우도
        
        V = intercept + β·X + λ_main·LV_main + Σ(λ_mod_i · LV_main · LV_mod_i)
        """
        # 파라미터 추출
        intercept = params['intercept']
        beta = params['beta']
        lambda_main = params['lambda_main']
        lambda_mod = params.get('lambda_mod', np.zeros(self.n_moderators))
        
        # 데이터 추출
        X = data[self.choice_attributes].values
        choice = data['choice'].values
        
        # 주 잠재변수
        lv_main = latent_vars[self.main_lv]
        
        # 효용 계산 - 기본 부분
        V = intercept + X @ beta + lambda_main * lv_main
        
        # 조절효과 추가
        for i, mod_lv_name in enumerate(self.moderator_lvs):
            lv_mod = latent_vars[mod_lv_name]
            # 상호작용항: LV_main × LV_mod
            interaction = lv_main * lv_mod
            V += lambda_mod[i] * interaction
        
        # 확률 계산
        prob_yes = norm.cdf(V)
        prob_yes = np.clip(prob_yes, 1e-10, 1 - 1e-10)
        
        # 로그우도
        ll = np.sum(
            choice * np.log(prob_yes) +
            (1 - choice) * np.log(1 - prob_yes)
        )
        
        return ll
    
    def initialize_parameters(self) -> Dict:
        """파라미터 초기화"""
        params = {
            'intercept': 0.0,
            'beta': np.zeros(self.n_attributes),
            'lambda_main': 1.0,
            'lambda_mod': np.zeros(self.n_moderators)
        }
        
        # 가격 변수 음수 초기화
        if self.config.price_variable in self.choice_attributes:
            price_idx = self.choice_attributes.index(self.config.price_variable)
            params['beta'][price_idx] = -1.0
        
        return params
```

---

## 📊 수치 예시

### 예시 1: 가격수준의 부적 조절효과

**파라미터**:
```python
lambda_main = 1.0
lambda_mod_price = -0.3
```

**시나리오**:
```
구매의도 (PI) = 1.0 (높음)

Case 1: 가격수준 (PP) = -1.0 (낮음, 저렴하다고 인식)
  V = ... + 1.0 × 1.0 + (-0.3) × (1.0 × -1.0)
    = ... + 1.0 + 0.3
    = ... + 1.3  ← 효용 증가!

Case 2: 가격수준 (PP) = 1.0 (높음, 비싸다고 인식)
  V = ... + 1.0 × 1.0 + (-0.3) × (1.0 × 1.0)
    = ... + 1.0 - 0.3
    = ... + 0.7  ← 효용 감소!
```

**해석**: 
- 구매의도가 같아도 가격수준이 높으면 실제 선택 확률 감소
- 가격이 장벽 역할

---

### 예시 2: 영양지식의 정적 조절효과

**파라미터**:
```python
lambda_main = 1.0
lambda_mod_knowledge = 0.2
```

**시나리오**:
```
구매의도 (PI) = 1.0 (높음)

Case 1: 영양지식 (NK) = -1.0 (낮음)
  V = ... + 1.0 × 1.0 + 0.2 × (1.0 × -1.0)
    = ... + 1.0 - 0.2
    = ... + 0.8  ← 효용 감소

Case 2: 영양지식 (NK) = 1.0 (높음)
  V = ... + 1.0 × 1.0 + 0.2 × (1.0 × 1.0)
    = ... + 1.0 + 0.2
    = ... + 1.2  ← 효용 증가!
```

**해석**: 
- 구매의도가 같아도 영양지식이 높으면 실제 선택 확률 증가
- 영양지식이 촉진 역할

---

### 예시 3: 복합 효과

**파라미터**:
```python
intercept = 0.0
beta = np.array([0.5, 0.3, -1.0])  # [sugar_free, health_label, price]
lambda_main = 1.0
lambda_mod_price = -0.3
lambda_mod_knowledge = 0.2
```

**시나리오**:
```
선택 속성: X = [1, 1, 0.5]  # sugar_free=1, health_label=1, price=0.5
구매의도: PI = 1.0
가격수준: PP = 0.5 (약간 비쌈)
영양지식: NK = 1.0 (높음)

V = 0.0 + 
    [1, 1, 0.5] @ [0.5, 0.3, -1.0] +  # 속성 효과
    1.0 × 1.0 +                        # 구매의도 주효과
    (-0.3) × (1.0 × 0.5) +             # 가격수준 조절
    0.2 × (1.0 × 1.0)                  # 영양지식 조절

  = 0.0 + (0.5 + 0.3 - 0.5) + 1.0 + (-0.15) + 0.2
  = 0.3 + 1.0 - 0.15 + 0.2
  = 1.35

P(choice=1) = Φ(1.35) ≈ 0.91 (91% 선택 확률)
```

---

## 🔍 조절효과 해석 방법

### Simple Slopes Analysis

조절변수의 수준별로 주효과를 계산:

```python
def calculate_simple_slopes(lambda_main, lambda_mod, moderator_values):
    """
    단순 기울기 계산
    
    Args:
        lambda_main: 주효과 계수
        lambda_mod: 조절효과 계수
        moderator_values: 조절변수 값 리스트 (예: [-1, 0, 1])
    
    Returns:
        각 조절변수 수준에서의 주효과
    """
    slopes = {}
    for mod_val in moderator_values:
        # 주효과 = λ_main + λ_mod × M
        slope = lambda_main + lambda_mod * mod_val
        slopes[f'M={mod_val}'] = slope
    
    return slopes

# 예시
slopes = calculate_simple_slopes(
    lambda_main=1.0,
    lambda_mod=-0.3,
    moderator_values=[-1, 0, 1]  # 낮음, 중간, 높음
)

print(slopes)
# {
#   'M=-1': 1.3,  # 가격수준 낮을 때: 구매의도 효과 강함
#   'M=0': 1.0,   # 가격수준 중간: 구매의도 효과 보통
#   'M=1': 0.7    # 가격수준 높을 때: 구매의도 효과 약함
# }
```

---

## 📈 시각화

### 조절효과 그래프

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_moderation_effect(lambda_main, lambda_mod, 
                          moderator_name='Moderator',
                          main_var_name='Main Variable'):
    """조절효과 시각화"""
    
    # 주 변수 범위
    main_var = np.linspace(-2, 2, 100)
    
    # 조절변수 수준 (낮음, 중간, 높음)
    mod_levels = {'Low (-1SD)': -1, 'Mean': 0, 'High (+1SD)': 1}
    
    plt.figure(figsize=(10, 6))
    
    for label, mod_val in mod_levels.items():
        # 효용 = λ_main × X + λ_mod × (X × M)
        #      = (λ_main + λ_mod × M) × X
        slope = lambda_main + lambda_mod * mod_val
        utility = slope * main_var
        
        plt.plot(main_var, utility, label=f'{moderator_name} {label}', linewidth=2)
    
    plt.xlabel(main_var_name, fontsize=12)
    plt.ylabel('Utility Contribution', fontsize=12)
    plt.title(f'Moderation Effect of {moderator_name}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    return plt

# 예시: 가격수준의 부적 조절효과
plot_moderation_effect(
    lambda_main=1.0,
    lambda_mod=-0.3,
    moderator_name='Perceived Price',
    main_var_name='Purchase Intention'
)
plt.show()
```

---

## ⚙️ 실제 추정 시 고려사항

### 1. 다중공선성 (Multicollinearity)

**문제**: 상호작용항 (PI × PP)이 주효과 (PI, PP)와 높은 상관

**해결책**: 중심화 (Centering)
```python
# 잠재변수 중심화
lv_main_centered = lv_main - np.mean(lv_main)
lv_mod_centered = lv_mod - np.mean(lv_mod)

# 상호작용항 계산
interaction = lv_main_centered * lv_mod_centered
```

**참고**: ICLV에서는 잠재변수가 이미 표준화되어 있으므로 (평균≈0) 중심화 불필요

---

### 2. 파라미터 초기값

```python
params = {
    'intercept': 0.0,
    'beta': np.zeros(n_attributes),
    'lambda_main': 1.0,      # 주효과: 양수로 초기화
    'lambda_mod': np.zeros(n_moderators)  # 조절효과: 0으로 초기화
}
```

---

### 3. 통계적 유의성 검정

조절효과가 유의한지 확인:
```python
# H0: λ_mod = 0 (조절효과 없음)
# H1: λ_mod ≠ 0 (조절효과 있음)

z_score = lambda_mod / se_lambda_mod
p_value = 2 * (1 - norm.cdf(abs(z_score)))

if p_value < 0.05:
    print("조절효과 유의함!")
```

---

## 🎯 요약

### 조절효과 구현 핵심

1. **입력 변경**: `lv` (스칼라) → `latent_vars` (딕셔너리)
2. **파라미터 추가**: `lambda` → `lambda_main` + `lambda_mod`
3. **효용 함수 수정**: 상호작용항 추가
   ```python
   V = intercept + β·X + λ_main·LV_main + λ_mod·(LV_main × LV_mod)
   ```

### 예상 결과

```
가격수준 조절효과: λ₂ ≈ -0.2 ~ -0.4 (부적, 장벽 역할)
영양지식 조절효과: λ₃ ≈ 0.2 ~ 0.4 (정적, 촉진 역할)
```

### 해석

- **가격수준 높음**: 구매의도 있어도 선택 확률 감소 → 가격 할인 필요
- **영양지식 높음**: 구매의도가 선택으로 전환 잘됨 → 교육 프로그램 효과적

---

## 📚 참고자료

- Aiken & West (1991). *Multiple Regression: Testing and Interpreting Interactions*
- Hayes (2018). *Introduction to Mediation, Moderation, and Conditional Process Analysis*
- Cohen et al. (2003). *Applied Multiple Regression/Correlation Analysis*

