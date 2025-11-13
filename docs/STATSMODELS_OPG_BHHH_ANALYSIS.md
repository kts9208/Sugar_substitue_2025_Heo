# Statsmodels OPG를 활용한 BHHH 구현 가능성 분석

**작성일**: 2025-11-13  
**작성자**: Taeseok Kim  
**목적**: Statsmodels의 OPG 기능을 활용하여 BHHH 최적화를 구현할 수 있는지 검토

---

## 📋 **요약**

### ✅ **결론: 가능하지만 GPU 가속과 충돌**

**Statsmodels OPG 활용**:
- ✅ `score_obs()` 메서드로 개인별 gradient 계산 가능
- ✅ `cov_type='opg'` 옵션으로 OPG 공분산 행렬 자동 계산
- ❌ **GPU 가속과 호환 불가능** (CPU 기반 scipy.optimize만 지원)
- ❌ **현재 GPU 구현을 포기해야 함**

**권장 사항**:
- 🎯 **현재 자체 구현 유지** (GPU 가속 활용)
- 🎯 **Statsmodels는 참고용으로만 사용**

---

## 🔍 **1. Statsmodels OPG 기능 분석**

### **1.1. `score_obs()` 메서드**

**역할**: 개인별 (observation-level) gradient 계산

**Statsmodels 구조**:
```python
from statsmodels.base.model import GenericLikelihoodModel

class MyModel(GenericLikelihoodModel):
    
    def loglike(self, params):
        """전체 log-likelihood"""
        return np.sum(self.loglikeobs(params))
    
    def loglikeobs(self, params):
        """개인별 log-likelihood (n_obs,)"""
        # 각 개인의 log-likelihood 반환
        return individual_ll  # shape: (n_obs,)
    
    def score_obs(self, params):
        """개인별 gradient (n_obs, n_params)"""
        # 각 개인의 gradient 반환
        return individual_gradients  # shape: (n_obs, n_params)
```

**OPG 공분산 행렬 계산**:
```python
# Fit with OPG covariance
results = model.fit(cov_type='opg')

# OPG 공분산 행렬 자동 계산
# Cov = inv(Σ_i g_i × g_i^T)
print(results.cov_params())  # OPG 기반 공분산
print(results.bse)  # OPG 기반 표준오차
```

---

## 🎯 **2. ICLV 모델에 적용 가능성**

### **2.1. 필요한 구현**

#### **Step 1: GenericLikelihoodModel 상속**

```python
from statsmodels.base.model import GenericLikelihoodModel
import numpy as np

class ICLVModel(GenericLikelihoodModel):
    """
    ICLV 모델을 Statsmodels GenericLikelihoodModel로 구현
    """
    
    def __init__(self, data, measurement_model, structural_model, 
                 choice_model, halton_generator, **kwargs):
        # 데이터는 개인 ID로 그룹화
        self.individual_ids = data['person_id'].unique()
        self.n_individuals = len(self.individual_ids)
        
        # 모델 저장
        self.measurement_model = measurement_model
        self.structural_model = structural_model
        self.choice_model = choice_model
        self.halton_generator = halton_generator
        self.data = data
        
        # GenericLikelihoodModel 초기화
        # endog는 더미 (ICLV는 복잡한 구조)
        endog = np.zeros(self.n_individuals)
        super(ICLVModel, self).__init__(endog, **kwargs)
    
    def loglikeobs(self, params):
        """
        개인별 log-likelihood 계산
        
        Args:
            params: 파라미터 벡터 (1D array)
        
        Returns:
            개인별 log-likelihood (n_individuals,)
        """
        # 파라미터 언팩
        param_dict = self._unpack_parameters(params)
        
        # 개인별 log-likelihood 계산
        ll_individuals = np.zeros(self.n_individuals)
        
        for i, ind_id in enumerate(self.individual_ids):
            # 개인 데이터
            ind_data = self.data[self.data['person_id'] == ind_id]
            ind_draws = self.halton_generator.get_draws()[i]
            
            # 개인 log-likelihood 계산 (Monte Carlo 적분)
            ll_individuals[i] = self._compute_individual_likelihood(
                ind_data, ind_draws, param_dict
            )
        
        return ll_individuals  # shape: (n_individuals,)
    
    def score_obs(self, params):
        """
        개인별 gradient 계산
        
        Args:
            params: 파라미터 벡터 (1D array)
        
        Returns:
            개인별 gradient (n_individuals, n_params)
        """
        # 파라미터 언팩
        param_dict = self._unpack_parameters(params)
        
        n_params = len(params)
        gradients = np.zeros((self.n_individuals, n_params))
        
        for i, ind_id in enumerate(self.individual_ids):
            # 개인 데이터
            ind_data = self.data[self.data['person_id'] == ind_id]
            ind_draws = self.halton_generator.get_draws()[i]
            
            # 개인별 gradient 계산
            grad_dict = self._compute_individual_gradient(
                ind_data, ind_draws, param_dict
            )
            
            # Gradient 벡터로 변환
            gradients[i, :] = self._pack_gradient(grad_dict)
        
        return gradients  # shape: (n_individuals, n_params)
```

#### **Step 2: Fit with OPG**

```python
# 모델 생성
iclv_model = ICLVModel(
    data, measurement_model, structural_model, 
    choice_model, halton_generator
)

# 초기 파라미터
initial_params = iclv_model._pack_parameters(initial_param_dict)

# Fit (scipy.optimize 사용)
results = iclv_model.fit(
    start_params=initial_params,
    method='bfgs',  # 또는 'newton', 'ncg'
    cov_type='opg'  # OPG 공분산 행렬
)

# 결과
print(results.params)  # 추정 파라미터
print(results.bse)  # OPG 기반 표준오차
print(results.cov_params())  # OPG 공분산 행렬
```

---

## ⚠️ **3. 문제점: GPU 가속과 충돌**

### **3.1. Statsmodels의 한계**

**Statsmodels는 CPU 기반**:
```python
# statsmodels.base.model.GenericLikelihoodModel.fit()
def fit(self, start_params=None, method='newton', ...):
    # scipy.optimize 사용 (CPU only)
    from scipy.optimize import minimize
    
    result = minimize(
        fun=lambda p: -self.loglike(p),
        x0=start_params,
        jac=lambda p: -self.score(p),  # CPU gradient
        method=method,
        ...
    )
```

**GPU 가속 불가능**:
- ❌ `loglikeobs()` 내부에서 GPU 사용 가능하지만
- ❌ Statsmodels 프레임워크는 CPU 기반 scipy.optimize만 지원
- ❌ GPU 배치 처리의 이점을 완전히 활용 불가

---

### **3.2. 현재 GPU 구현과 비교**

| 측면 | 현재 자체 구현 | Statsmodels OPG |
|------|---------------|-----------------|
| **GPU 가속** | ✅ CuPy 배치 처리 | ❌ CPU only |
| **개인별 gradient** | ✅ GPU 배치 계산 | ❌ CPU 순차 계산 |
| **메모리 효율** | ✅ GPU 메모리 활용 | ❌ CPU 메모리만 |
| **속도** | ✅ 매우 빠름 | ❌ 느림 |
| **OPG 계산** | ✅ 자체 구현 | ✅ 자동 계산 |
| **유연성** | ✅ 완전 제어 | ❌ 프레임워크 제약 |

**성능 비교 예상**:
```
현재 GPU 구현: 개인별 gradient 계산 90초 (GPU 배치)
Statsmodels: 개인별 gradient 계산 ~30분 (CPU 순차)
→ 약 20배 느림
```

---

## 🔧 **4. 하이브리드 접근법**

### **4.1. GPU 계산 + Statsmodels OPG 검증**

**아이디어**: GPU로 계산하고 Statsmodels로 검증

```python
# 1. 현재 GPU 구현으로 추정
estimator = SimultaneousEstimator(config)
results_gpu = estimator.estimate(data, measurement_model, 
                                  structural_model, choice_model)

# 2. Statsmodels로 검증 (작은 샘플)
# 소수의 개인만 사용하여 OPG 계산 검증
sample_data = data[data['person_id'].isin(sample_ids)]
iclv_model = ICLVModel(sample_data, ...)
results_sm = iclv_model.fit(
    start_params=results_gpu['parameters'],
    cov_type='opg'
)

# 3. 표준오차 비교
print("GPU BHHH SE:", results_gpu['standard_errors'])
print("Statsmodels OPG SE:", results_sm.bse)
print("차이:", np.abs(results_gpu['standard_errors'] - results_sm.bse))
```

---

## 📊 **5. 구현 복잡도 비교**

### **5.1. Statsmodels OPG 사용**

**장점**:
- ✅ OPG 공분산 자동 계산
- ✅ 표준오차, t-통계량, p-값 자동 제공
- ✅ 검증된 프레임워크

**단점**:
- ❌ GPU 가속 불가능
- ❌ 20배 이상 느림
- ❌ 프레임워크 제약 (유연성 낮음)
- ❌ ICLV 복잡한 구조 구현 어려움

**구현 난이도**: ⭐⭐⭐⭐ (높음)
- `GenericLikelihoodModel` 상속
- `loglikeobs()` 구현 (개인별 LL)
- `score_obs()` 구현 (개인별 gradient)
- 파라미터 pack/unpack
- Monte Carlo 적분 통합

---

### **5.2. 현재 자체 구현 유지**

**장점**:
- ✅ GPU 가속 (20배 빠름)
- ✅ 완전한 제어
- ✅ 이미 구현 완료
- ✅ BHHH OPG 계산 구현됨

**단점**:
- ⚠️ 자체 검증 필요

**구현 난이도**: ⭐ (이미 완료)

---

## 🎯 **6. 최종 권장 사항**

### **✅ 현재 자체 구현 유지**

**이유**:
1. **GPU 가속 필수**: 20배 성능 차이
2. **이미 구현 완료**: BHHH OPG 계산 모듈 완성
3. **유연성**: 완전한 제어 가능
4. **확장성**: 향후 개선 용이

### **📚 Statsmodels는 참고용**

**활용 방법**:
1. **이론 검증**: OPG 계산 방식 확인
2. **소규모 검증**: 작은 샘플로 결과 비교
3. **문서화**: 표준 방법론 참조

---

## 💡 **7. 대안: Statsmodels 스타일 인터페이스**

### **7.1. Statsmodels 스타일 래퍼 생성**

현재 구현을 Statsmodels 스타일로 래핑:

```python
class ICLVModelWrapper:
    """
    Statsmodels 스타일 인터페이스
    내부는 GPU 가속 사용
    """
    
    def __init__(self, data, config, measurement_model, 
                 structural_model, choice_model):
        self.estimator = SimultaneousEstimator(config)
        self.data = data
        self.measurement_model = measurement_model
        self.structural_model = structural_model
        self.choice_model = choice_model
    
    def fit(self, cov_type='bhhh'):
        """
        Statsmodels 스타일 fit 메서드
        """
        # GPU 가속 추정
        results = self.estimator.estimate(
            self.data, 
            self.measurement_model,
            self.structural_model,
            self.choice_model
        )
        
        # Statsmodels 스타일 결과 객체 반환
        return ICLVResults(results, cov_type=cov_type)

class ICLVResults:
    """Statsmodels 스타일 결과 객체"""
    
    def __init__(self, results, cov_type='bhhh'):
        self.params = results['parameters']
        self.bse = results['standard_errors']
        self.tvalues = results['t_statistics']
        self.pvalues = results['p_values']
        self.cov_params_matrix = results['hessian_inv']
        self.llf = results['log_likelihood']
        self.aic = results['aic']
        self.bic = results['bic']
    
    def summary(self):
        """결과 요약 출력"""
        # Statsmodels 스타일 요약 테이블
        pass
```

**사용 예시**:
```python
# Statsmodels 스타일 사용
model = ICLVModelWrapper(data, config, measurement_model, 
                         structural_model, choice_model)
results = model.fit(cov_type='bhhh')

# Statsmodels 스타일 결과 접근
print(results.params)
print(results.bse)
print(results.summary())
```

---

## 📝 **8. 결론**

### **Statsmodels OPG 활용 가능성**

| 측면 | 평가 | 비고 |
|------|------|------|
| **기술적 가능성** | ✅ 가능 | `score_obs()` 구현 필요 |
| **GPU 호환성** | ❌ 불가능 | CPU only |
| **성능** | ❌ 느림 | 20배 이상 느림 |
| **구현 복잡도** | ⭐⭐⭐⭐ | 높음 |
| **권장 여부** | ❌ 비권장 | GPU 가속 포기 |

### **최종 결론**

**❌ Statsmodels OPG 직접 사용: 비권장**
- GPU 가속 불가능
- 20배 이상 느림
- 현재 구현이 우수

**✅ 현재 자체 구현 유지: 강력 권장**
- GPU 가속 활용
- 이미 BHHH OPG 구현 완료
- 높은 성능 및 유연성

**📚 Statsmodels 활용 방법**
- 이론 검증 및 참고용
- 소규모 샘플 검증
- Statsmodels 스타일 래퍼 생성 (선택사항)

---

**결론**: Statsmodels의 OPG 기능은 기술적으로 활용 가능하지만, **GPU 가속과 충돌**하므로 현재 자체 구현을 유지하는 것이 최선입니다.

