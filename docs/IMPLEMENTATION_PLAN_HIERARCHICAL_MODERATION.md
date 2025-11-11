# 계층적 구조 및 조절효과 ICLV 모델 구현 계획서

**작성일**: 2025-11-11  
**담당**: Sugar Substitute Research Team  
**목적**: 기존 ICLV 모델을 계층적 구조와 조절효과를 포함한 새로운 모델로 전환

---

## 📋 Executive Summary

### 구현 목표
기존 병렬 구조 ICLV 모델을 **계층적 구조 + 조절효과** 모델로 전환하여:
1. 이론적 타당성 강화
2. 간접효과 분석 가능
3. 의도-행동 간극 설명력 향상

### 주요 변경사항

| 항목 | 현재 | 변경 후 |
|------|------|---------|
| **구조모델** | 4개 외생 LV → 구매의도 (병렬) | 건강관심도 → 건강유익성 → 구매의도 (계층) |
| **선택모델** | V = β·X + λ·PI | V = β·X + λ₁·PI + λ₂·(PI×PP) + λ₃·(PI×NK) |
| **사회인구학적 변수** | 구매의도에 직접 효과 | 제거 |
| **파라미터 수** | 12개 (구조 7 + 선택 5) | 9개 (구조 2 + 선택 7) |

### 예상 소요 시간
- **Phase 1**: 4-6시간 (계층적 구조모델)
- **Phase 2**: 2-3시간 (조절효과 선택모델)
- **Phase 3**: 4-6시간 (통합 및 테스트)
- **Phase 4**: 3-4시간 (결과 분석 및 문서화)
- **총 예상**: **13-19시간**

---

## 🎯 Phase 1: 계층적 구조모델 구현

### 목표
건강관심도 → 건강유익성 → 구매의도 계층 구조 구현

### 1.1 HierarchicalPath 데이터클래스 생성

**파일**: `src/analysis/hybrid_choice_model/iclv_models/hierarchical_config.py`

**작업 내용**:
```python
@dataclass
class HierarchicalPath:
    """계층적 경로 정의"""
    target: str                    # 목표 잠재변수
    predictors: List[str]          # 예측 잠재변수들
    error_variance: float = 1.0
    fix_error_variance: bool = True
```

**예상 시간**: 30분

---

### 1.2 HierarchicalStructuralConfig 클래스 생성

**파일**: `src/analysis/hybrid_choice_model/iclv_models/hierarchical_config.py`

**작업 내용**:
```python
@dataclass
class HierarchicalStructuralConfig:
    """계층적 구조모델 설정"""
    first_order_lvs: List[str]              # 1차 LV (외생)
    hierarchical_paths: List[HierarchicalPath]  # 계층 경로
    covariates: List[str] = field(default_factory=list)
    
    def get_n_parameters(self) -> int:
        """파라미터 수 계산"""
    
    def get_parameter_names(self) -> List[str]:
        """파라미터 이름 리스트"""
```

**검증 로직**:
- 순환 경로 방지
- 모든 predictor가 정의되어 있는지 확인
- target이 first_order_lvs에 없는지 확인

**예상 시간**: 1시간

---

### 1.3 HierarchicalStructural 클래스 생성

**파일**: `src/analysis/hybrid_choice_model/iclv_models/hierarchical_structural.py`

**작업 내용**:

#### **핵심 메서드 1: predict()**
```python
def predict(self, data: pd.DataFrame,
            first_order_draws: np.ndarray,
            params: Dict[str, float],
            higher_order_draws: Dict[str, float]) -> Dict[str, float]:
    """
    계층적 잠재변수 예측
    
    순서:
    1. 1차 LV = draws (외생)
    2. 2차 LV = f(1차 LV) + error
    3. 3차 LV = f(2차 LV) + error
    """
```

**구현 로직**:
```python
latent_vars = {}

# 1차 LV (외생)
for i, lv_name in enumerate(self.first_order_lvs):
    latent_vars[lv_name] = first_order_draws[i]

# 2차+ LV (내생) - 순서대로 계산
for path in self.hierarchical_paths:
    lv_mean = 0.0
    for pred in path.predictors:
        param_name = f'gamma_{pred}_to_{path.target}'
        gamma = params[param_name]
        lv_mean += gamma * latent_vars[pred]
    
    error_draw = higher_order_draws[path.target]
    latent_vars[path.target] = lv_mean + np.sqrt(path.error_variance) * error_draw

return latent_vars
```

#### **핵심 메서드 2: log_likelihood()**
```python
def log_likelihood(self, data: pd.DataFrame,
                  latent_vars: Dict[str, float],
                  first_order_draws: np.ndarray,
                  params: Dict[str, float],
                  higher_order_draws: Dict[str, float]) -> float:
    """
    계층적 구조모델 로그우도
    
    LL = Σ log P(LV_1st) + Σ log P(LV_higher | LV_predictors)
    """
```

**예상 시간**: 2-3시간

---

### 1.4 계층적 구조모델 단위 테스트

**파일**: `tests/test_hierarchical_structural.py`

**테스트 케이스**:
1. **Config 검증 테스트**
   - 순환 경로 감지
   - 정의되지 않은 predictor 감지

2. **Predict 테스트**
   - 1차 LV 값 확인
   - 2차 LV 계산 정확성
   - 3차 LV 계산 정확성

3. **Log-likelihood 테스트**
   - 수치 정확성 확인
   - 경계 조건 테스트

**예상 시간**: 1-1.5시간

---

## 🎯 Phase 2: 조절효과 선택모델 구현

### 목표
가격수준과 영양지식이 구매의도 → 선택 관계를 조절하는 모델 구현

### 2.1 BinaryProbitChoiceWithModeration 클래스 생성

**파일**: `src/analysis/hybrid_choice_model/iclv_models/choice_with_moderation.py`

**작업 내용**:

#### **핵심 메서드: log_likelihood()**
```python
def log_likelihood(self, data: pd.DataFrame,
                  latent_vars: Dict[str, float],
                  params: Dict) -> float:
    """
    조절효과 포함 로그우도
    
    V = intercept + β·X + λ_main·LV_main + Σ(λ_mod_i · LV_main · LV_mod_i)
    """
    intercept = params['intercept']
    beta = params['beta']
    lambda_main = params['lambda_main']
    lambda_mod = params.get('lambda_mod', np.zeros(self.n_moderators))
    
    X = data[self.choice_attributes].values
    choice = data['choice'].values
    
    lv_main = latent_vars[self.main_lv]
    
    # 효용 계산
    V = intercept + X @ beta + lambda_main * lv_main
    
    # 조절효과 추가
    for i, mod_lv_name in enumerate(self.moderator_lvs):
        lv_mod = latent_vars[mod_lv_name]
        V += lambda_mod[i] * (lv_main * lv_mod)  # 상호작용항
    
    # 확률 및 로그우도
    prob_yes = norm.cdf(V)
    prob_yes = np.clip(prob_yes, 1e-10, 1 - 1e-10)
    
    ll = np.sum(choice * np.log(prob_yes) + (1 - choice) * np.log(1 - prob_yes))
    return ll
```

**예상 시간**: 1.5-2시간

---

### 2.2 조절효과 선택모델 단위 테스트

**파일**: `tests/test_choice_with_moderation.py`

**테스트 케이스**:
1. **기본 기능 테스트**
   - 조절효과 없을 때 기존 모델과 동일한지 확인
   
2. **조절효과 테스트**
   - 부적 조절 (λ_mod < 0) 동작 확인
   - 정적 조절 (λ_mod > 0) 동작 확인

3. **수치 예시 검증**
   - 문서의 예시와 동일한 결과 확인

**예상 시간**: 1시간

---

### 2.3 ChoiceConfig 확장

**파일**: `src/analysis/hybrid_choice_model/iclv_models/iclv_config.py`

**작업 내용**:
```python
@dataclass
class ChoiceConfig:
    """선택모델 설정"""
    choice_attributes: List[str]
    choice_type: str = 'binary'
    price_variable: str = 'price'
    
    # 조절효과 설정 (NEW)
    moderation_enabled: bool = False
    moderator_lvs: Optional[List[str]] = None
    main_lv: str = 'purchase_intention'
```

**예상 시간**: 30분

---

## 🎯 Phase 3: 통합 및 테스트

### 3.1 HierarchicalMultiLatentConfig 통합 클래스 생성

**파일**: `src/analysis/hybrid_choice_model/iclv_models/hierarchical_multi_latent_config.py`

**작업 내용**:
```python
@dataclass
class HierarchicalMultiLatentConfig:
    """계층적 구조 + 조절효과 통합 설정"""
    
    measurement_configs: Dict[str, MeasurementConfig]
    structural: HierarchicalStructuralConfig
    choice: ChoiceConfig
    estimation: EstimationConfig
    
    individual_id_column: str = 'respondent_id'
    choice_column: str = 'choice'
```

**예상 시간**: 1시간

---

### 3.2 GPUBatchEstimator 수정

**파일**: `src/analysis/hybrid_choice_model/iclv_models/gpu_batch_estimator.py`

**수정 내용**:
1. **입력 처리 변경**
   - `lv` (스칼라) → `latent_vars` (딕셔너리)

2. **파라미터 구조 변경**
   - 계층적 구조 파라미터 처리
   - 조절효과 파라미터 처리

3. **로그우도 계산 수정**
   - 새로운 구조모델 호출
   - 새로운 선택모델 호출

**예상 시간**: 2-3시간

---

### 3.3 시뮬레이션 데이터 테스트

**파일**: `tests/test_hierarchical_moderation_simulation.py`

**작업 내용**:
1. **데이터 생성**
   - 알려진 파라미터로 시뮬레이션 데이터 생성
   - 계층 구조 반영
   - 조절효과 반영

2. **파라미터 복원**
   - 추정된 파라미터가 실제 값과 일치하는지 확인
   - 허용 오차 범위 설정 (±10%)

**예상 시간**: 2시간

---

### 3.4 실제 데이터 추정 테스트

**파일**: `scripts/test_hierarchical_moderation_iclv.py`

**작업 내용**:
1. **데이터 로드**
   - `data/processed/iclv/integrated_data.csv`

2. **모델 설정**
   ```python
   structural_config = HierarchicalStructuralConfig(
       first_order_lvs=['health_concern', 'perceived_price', 'nutrition_knowledge'],
       hierarchical_paths=[
           HierarchicalPath(
               target='perceived_benefit',
               predictors=['health_concern'],
               error_variance=1.0
           ),
           HierarchicalPath(
               target='purchase_intention',
               predictors=['perceived_benefit'],
               error_variance=1.0
           )
       ],
       covariates=[]  # 사회인구학적 변수 제거
   )
   
   choice_config = ChoiceConfig(
       choice_attributes=['sugar_free', 'health_label', 'price'],
       moderation_enabled=True,
       moderator_lvs=['perceived_price', 'nutrition_knowledge'],
       main_lv='purchase_intention'
   )
   ```

3. **추정 실행**
   - 수렴성 확인
   - 파라미터 유의성 확인

**예상 시간**: 1-2시간

---

### 3.5 기존 모델과 비교

**파일**: `scripts/compare_models.py`

**비교 항목**:
1. **적합도 지표**
   - Log-likelihood
   - AIC
   - BIC

2. **파라미터 수**
   - 현재: 12개
   - 새 모델: 9개

3. **해석력**
   - 간접효과 분석 가능 여부
   - 조절효과 해석 가능 여부

**예상 시간**: 1시간

---

## 🎯 Phase 4: 결과 분석 및 문서화

### 4.1 간접효과 분석

**파일**: `scripts/analyze_indirect_effects.py`

**분석 내용**:
```python
# 간접효과 계산
gamma_hc_to_pb = params['gamma_health_concern_to_perceived_benefit']
gamma_pb_to_pi = params['gamma_perceived_benefit_to_purchase_intention']

indirect_effect = gamma_hc_to_pb * gamma_pb_to_pi

print(f"건강관심도 → 건강유익성 → 구매의도 간접효과: {indirect_effect:.4f}")
```

**예상 시간**: 1시간

---

### 4.2 조절효과 해석

**파일**: `scripts/analyze_moderation_effects.py`

**분석 내용**:
1. **Simple Slopes Analysis**
   ```python
   # 가격수준 수준별 구매의도 효과
   for price_level in [-1, 0, 1]:
       slope = lambda_main + lambda_mod_price * price_level
       print(f"가격수준 {price_level}: 구매의도 효과 = {slope:.3f}")
   ```

2. **시각화**
   - 조절효과 그래프
   - 확률 표면 3D 그래프

**예상 시간**: 1.5시간

---

### 4.3 최종 결과 보고서 작성

**파일**: `results/HIERARCHICAL_MODERATION_RESULTS.md`

**포함 내용**:
1. **모델 개요**
2. **파라미터 추정 결과**
3. **간접효과 분석**
4. **조절효과 해석**
5. **기존 모델과 비교**
6. **실무적 시사점**

**예상 시간**: 1.5시간

---

### 4.4 시각화 자료 생성

**파일**: `scripts/create_visualizations.py`

**생성 자료**:
1. **경로 다이어그램** (Mermaid)
2. **조절효과 그래프** (Matplotlib)
3. **확률 표면** (3D plot)
4. **파라미터 추정 결과 표** (CSV)

**예상 시간**: 1시간

---

## 📊 구현 일정

### Week 1
- **Day 1-2**: Phase 1 (계층적 구조모델)
- **Day 3**: Phase 2 (조절효과 선택모델)

### Week 2
- **Day 4-5**: Phase 3 (통합 및 테스트)
- **Day 6**: Phase 4 (결과 분석 및 문서화)

---

## ✅ 체크리스트

### Phase 1: 계층적 구조모델
- [ ] HierarchicalPath 데이터클래스 생성
- [ ] HierarchicalStructuralConfig 클래스 생성
- [ ] HierarchicalStructural 클래스 생성
- [ ] 단위 테스트 작성 및 통과

### Phase 2: 조절효과 선택모델
- [ ] BinaryProbitChoiceWithModeration 클래스 생성
- [ ] 단위 테스트 작성 및 통과
- [ ] ChoiceConfig 확장

### Phase 3: 통합 및 테스트
- [ ] HierarchicalMultiLatentConfig 통합 클래스 생성
- [ ] GPUBatchEstimator 수정
- [ ] 시뮬레이션 데이터 테스트 통과
- [ ] 실제 데이터 추정 성공
- [ ] 기존 모델과 비교 완료

### Phase 4: 결과 분석 및 문서화
- [ ] 간접효과 분석 완료
- [ ] 조절효과 해석 완료
- [ ] 최종 결과 보고서 작성
- [ ] 시각화 자료 생성

---

## 🎯 성공 기준

1. **기술적 성공**
   - 모든 단위 테스트 통과
   - 실제 데이터 추정 수렴
   - 파라미터 유의성 확인

2. **이론적 성공**
   - 간접효과 유의함 (p < 0.05)
   - 조절효과 유의함 (p < 0.05)
   - 예상 방향과 일치 (가격수준: 부적, 영양지식: 정적)

3. **모델 적합도**
   - AIC/BIC가 기존 모델보다 낮음 (또는 유사)
   - 로그우도 증가

---

## 📝 리스크 및 대응 방안

### 리스크 1: 수렴 실패
**대응**: 
- 초기값 조정
- 최적화 알고리즘 변경 (BFGS → L-BFGS-B)
- 파라미터 제약 조건 추가

### 리스크 2: 조절효과 비유의
**대응**:
- 표본 크기 확인
- 조절변수 재선택
- 이론적 근거 재검토

### 리스크 3: 모델 적합도 저하
**대응**:
- 사회인구학적 변수 재추가 고려
- 다른 계층 구조 시도
- 기존 모델 유지 및 대안 모델로 보고

---

## 📚 참고 자료

- `docs/STRUCTURAL_MODEL_MODIFICATION_FEASIBILITY.md`
- `docs/HIERARCHICAL_MODERATION_IMPLEMENTATION_GUIDE.md`
- `docs/THEORETICAL_IMPLICATIONS_HIERARCHICAL_MODEL.md`
- `docs/MODERATION_EFFECT_IMPLEMENTATION_DETAILED.md`
- `examples/moderation_effect_example.py`

---

## 🎉 기대 효과

1. **이론적 기여**: 계층적 메커니즘 규명
2. **방법론적 기여**: ICLV 모델 확장
3. **실무적 기여**: 마케팅 전략 수립 근거 제공

