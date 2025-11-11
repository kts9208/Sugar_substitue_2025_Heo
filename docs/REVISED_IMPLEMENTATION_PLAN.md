# 기존 ICLV 모델 수정 계획서 (REVISED)

**작성일**: 2025-11-11  
**목적**: 기존 클래스들을 수정하여 계층적 구조 + 조절효과 구현

---

## 🎯 핵심 변경사항

**새로운 클래스 생성 ❌**  
**기존 클래스 수정 ✅**

---

## 📊 수정할 기존 클래스

### 1. `MultiLatentStructuralConfig` (설정 클래스)
**파일**: `src/analysis/hybrid_choice_model/iclv_models/multi_latent_config.py`

**현재 구조**:
```python
@dataclass
class MultiLatentStructuralConfig:
    endogenous_lv: str                    # 'purchase_intention'
    exogenous_lvs: List[str]              # 4개 외생 LV
    covariates: List[str]                 # 사회인구학적 변수
    error_variance: float = 1.0
```

**수정 후**:
```python
@dataclass
class MultiLatentStructuralConfig:
    endogenous_lv: str
    exogenous_lvs: List[str]
    covariates: List[str]
    error_variance: float = 1.0
    
    # ✅ 새로 추가
    hierarchical_paths: Optional[List[Dict[str, Any]]] = None
    # 예: [{'target': 'perceived_benefit', 'predictors': ['health_concern']},
    #      {'target': 'purchase_intention', 'predictors': ['perceived_benefit']}]
```

---

### 2. `MultiLatentStructural` (구조모델 클래스)
**파일**: `src/analysis/hybrid_choice_model/iclv_models/multi_latent_structural.py`

**현재 로직**:
```python
# 외생 LV: 표준정규분포
for i, lv_name in enumerate(self.exogenous_lvs):
    latent_vars[lv_name] = exo_draws[i]

# 내생 LV: 모든 외생 LV의 선형결합
lv_mean = gamma_lv @ exo_draws + gamma_x @ X_values
latent_vars[self.endogenous_lv] = lv_mean + np.sqrt(self.error_variance) * endo_draw
```

**수정 후 로직**:
```python
# 1차 LV (외생): 표준정규분포
for lv_name in first_order_lvs:
    latent_vars[lv_name] = draws[lv_name]

# 계층적 경로 순서대로 계산
if self.config.hierarchical_paths:
    for path in self.config.hierarchical_paths:
        # 예: perceived_benefit = gamma * health_concern + error
        lv_mean = 0.0
        for pred in path['predictors']:
            param_name = f'gamma_{pred}_to_{path["target"]}'
            lv_mean += params[param_name] * latent_vars[pred]
        
        latent_vars[path['target']] = lv_mean + error_draw
else:
    # 기존 병렬 구조 (하위 호환성)
    lv_mean = gamma_lv @ exo_draws + gamma_x @ X_values
    latent_vars[self.endogenous_lv] = lv_mean + error_draw
```

---

### 3. `BinaryProbitChoice` (선택모델 클래스)
**파일**: `src/analysis/hybrid_choice_model/iclv_models/choice_equations.py`

**현재 로직**:
```python
def log_likelihood(self, data, lv, params):
    # lv: 스칼라 또는 배열 (구매의도만)
    V = intercept + X @ beta + lambda_lv * lv_array
    prob_yes = norm.cdf(V)
    ll = np.sum(choice * np.log(prob_yes) + (1-choice) * np.log(1-prob_yes))
```

**수정 후 로직**:
```python
def log_likelihood(self, data, lv, params):
    # lv: 스칼라, 배열, 또는 딕셔너리 (하위 호환성)
    
    # 딕셔너리인 경우 (조절효과 모드)
    if isinstance(lv, dict):
        lv_main = lv[self.config.main_lv]  # 구매의도
        
        # 기본 효용
        V = intercept + X @ beta + params['lambda_main'] * lv_main
        
        # 조절효과 추가
        if self.config.moderation_enabled:
            for mod_lv_name in self.config.moderator_lvs:
                lv_mod = lv[mod_lv_name]
                param_name = f'lambda_mod_{mod_lv_name}'
                V += params[param_name] * (lv_main * lv_mod)
    
    # 스칼라/배열인 경우 (기존 방식, 하위 호환성)
    else:
        V = intercept + X @ beta + params['lambda'] * lv_array
    
    prob_yes = norm.cdf(V)
    ll = np.sum(choice * np.log(prob_yes) + (1-choice) * np.log(1-prob_yes))
```

---

### 4. `ChoiceConfig` (선택모델 설정)
**파일**: `src/analysis/hybrid_choice_model/iclv_models/iclv_config.py`

**현재 구조**:
```python
@dataclass
class ChoiceConfig:
    choice_attributes: List[str]
    choice_type: str = 'binary'
    price_variable: str = 'price'
```

**수정 후**:
```python
@dataclass
class ChoiceConfig:
    choice_attributes: List[str]
    choice_type: str = 'binary'
    price_variable: str = 'price'
    
    # ✅ 새로 추가
    moderation_enabled: bool = False
    moderator_lvs: Optional[List[str]] = None
    main_lv: str = 'purchase_intention'
```

---

## 🔧 구현 단계

### Phase 1: 설정 클래스 수정 (1시간)

#### Task 1.1: `MultiLatentStructuralConfig` 수정
- [ ] `hierarchical_paths` 필드 추가
- [ ] 검증 로직 추가 (순환 경로 방지)
- [ ] 파라미터 수 계산 로직 수정

#### Task 1.2: `ChoiceConfig` 수정
- [ ] `moderation_enabled` 필드 추가
- [ ] `moderator_lvs` 필드 추가
- [ ] `main_lv` 필드 추가

---

### Phase 2: 구조모델 수정 (3-4시간)

#### Task 2.1: `MultiLatentStructural.predict()` 수정
- [ ] 계층적 경로 처리 로직 추가
- [ ] 기존 병렬 구조 하위 호환성 유지
- [ ] 파라미터 이름 동적 생성 (`gamma_X_to_Y`)

#### Task 2.2: `MultiLatentStructural.log_likelihood()` 수정
- [ ] 계층적 구조 로그우도 계산
- [ ] 기존 방식 하위 호환성 유지

#### Task 2.3: `MultiLatentStructural.initialize_parameters()` 수정
- [ ] 계층적 경로 파라미터 초기화
- [ ] 파라미터 이름 매핑 생성

---

### Phase 3: 선택모델 수정 (2-3시간)

#### Task 3.1: `BinaryProbitChoice.log_likelihood()` 수정
- [ ] 입력 타입 체크 (스칼라/배열/딕셔너리)
- [ ] 조절효과 로직 추가
- [ ] 기존 방식 하위 호환성 유지

#### Task 3.2: `BinaryProbitChoice.predict_probabilities()` 수정
- [ ] 조절효과 반영
- [ ] 하위 호환성 유지

#### Task 3.3: `BinaryProbitChoice.initialize_parameters()` 수정
- [ ] 조절효과 파라미터 초기화
- [ ] `lambda` → `lambda_main` + `lambda_mod_*`

---

### Phase 4: Estimator 수정 (2-3시간)

#### Task 4.1: `GPUBatchEstimator` 수정
- [ ] 계층적 구조 파라미터 처리
- [ ] 조절효과 파라미터 처리
- [ ] 잠재변수 딕셔너리 전달 로직

#### Task 4.2: 파라미터 언패킹/패킹 수정
- [ ] `unpack_parameters()` 수정
- [ ] `pack_parameters()` 수정
- [ ] 파라미터 이름 매핑 업데이트

---

### Phase 5: 테스트 및 검증 (3-4시간)

#### Task 5.1: 단위 테스트
- [ ] 구조모델 테스트 (계층 vs 병렬)
- [ ] 선택모델 테스트 (조절효과 vs 기본)
- [ ] 하위 호환성 테스트

#### Task 5.2: 통합 테스트
- [ ] 시뮬레이션 데이터 테스트
- [ ] 실제 데이터 추정
- [ ] 기존 모델과 비교

---

## 📝 사용 예시

### 기존 방식 (하위 호환성 유지)
```python
# 기존 코드 그대로 작동
structural_config = MultiLatentStructuralConfig(
    endogenous_lv='purchase_intention',
    exogenous_lvs=['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge'],
    covariates=['age_std', 'gender', 'income_std'],
    error_variance=1.0
)

choice_config = ChoiceConfig(
    choice_attributes=['sugar_free', 'health_label', 'price']
)
```

### 새로운 방식 (계층 + 조절)
```python
# 계층적 구조 설정
structural_config = MultiLatentStructuralConfig(
    endogenous_lv='purchase_intention',
    exogenous_lvs=['health_concern', 'perceived_price', 'nutrition_knowledge'],
    covariates=[],  # 사회인구학적 변수 제거
    error_variance=1.0,
    hierarchical_paths=[
        {'target': 'perceived_benefit', 'predictors': ['health_concern']},
        {'target': 'purchase_intention', 'predictors': ['perceived_benefit']}
    ]
)

# 조절효과 설정
choice_config = ChoiceConfig(
    choice_attributes=['sugar_free', 'health_label', 'price'],
    moderation_enabled=True,
    moderator_lvs=['perceived_price', 'nutrition_knowledge'],
    main_lv='purchase_intention'
)
```

---

## ⏱️ 예상 소요 시간

```
Phase 1: 설정 수정        1시간    ████░░░░░░░░░░░░░░░░
Phase 2: 구조모델 수정    3-4시간  ████████████░░░░░░░░
Phase 3: 선택모델 수정    2-3시간  ████████░░░░░░░░░░░░
Phase 4: Estimator 수정   2-3시간  ████████░░░░░░░░░░░░
Phase 5: 테스트 및 검증   3-4시간  ████████████░░░░░░░░
─────────────────────────────────────────────────────
총 예상: 11-15시간 (약 2일)
```

---

## ✅ 하위 호환성 보장

모든 수정은 **하위 호환성**을 유지합니다:

1. **기존 코드 그대로 작동**
   - `hierarchical_paths=None` → 병렬 구조
   - `moderation_enabled=False` → 기본 선택모델

2. **점진적 마이그레이션 가능**
   - 구조모델만 먼저 수정 가능
   - 선택모델만 먼저 수정 가능

3. **기존 테스트 통과**
   - `test_gpu_batch_iclv.py` 그대로 실행 가능

---

## 🎯 성공 기준

1. **기존 코드 호환성**
   - ✅ `test_gpu_batch_iclv.py` 수정 없이 실행
   - ✅ 기존 결과와 동일한 추정값

2. **새로운 기능**
   - ✅ 계층적 구조 추정 성공
   - ✅ 조절효과 추정 성공
   - ✅ 파라미터 유의성 확인

3. **코드 품질**
   - ✅ 모든 단위 테스트 통과
   - ✅ 타입 힌트 유지
   - ✅ 문서화 완료

---

## 🚀 다음 단계

구현을 시작하시겠습니까?

**옵션 1**: Phase 1부터 순차적으로 구현  
**옵션 2**: 특정 Phase만 먼저 구현  
**옵션 3**: 계획 추가 검토

