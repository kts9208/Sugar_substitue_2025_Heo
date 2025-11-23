# Sign Correction & Alignment 설계 문서

## 📋 문제 정의

### 1. **Sign Indeterminacy (부호 불확정성)**

SEM에서 잠재변수는 **식별 제약(identification constraint)**을 위해 첫 번째 요인적재량을 1로 고정합니다. 하지만 이것만으로는 잠재변수의 **부호(sign)**가 결정되지 않습니다.

**예시:**
```
원본 추정:
  LV = 0.8 * X1 + 0.6 * X2 + 0.4 * X3
  
부호 반전된 추정 (동일한 적합도):
  LV = -0.8 * X1 - 0.6 * X2 - 0.4 * X3
```

두 모델은 **통계적으로 동일**하지만, 잠재변수의 부호가 반대입니다.

### 2. **부트스트랩에서의 문제**

부트스트랩 샘플마다 잠재변수의 부호가 **무작위로 반전**될 수 있습니다:

```
원본:      LV = +0.8 * X1 + 0.6 * X2
샘플 1:    LV = +0.7 * X1 + 0.5 * X2  ✅ 같은 부호
샘플 2:    LV = -0.9 * X1 - 0.7 * X2  ⚠️ 부호 반전!
샘플 3:    LV = +0.6 * X1 + 0.4 * X2  ✅ 같은 부호
```

**결과:**
- 평균 = (0.7 - 0.9 + 0.6) / 3 = **0.13** ❌ (실제 0.8과 매우 다름)
- 표준편차가 과도하게 커짐
- 신뢰구간이 0을 포함하여 비유의하게 나타남

---

## 🎯 해결 방안

### **방법 1: Factor Loading Sign Alignment (요인적재량 부호 정렬)**

**원리:** 각 부트스트랩 샘플의 요인적재량을 원본과 비교하여 부호를 정렬합니다.

**알고리즘:**
```python
def align_factor_loadings(original_loadings, bootstrap_loadings):
    """
    부트스트랩 요인적재량을 원본과 부호 정렬
    
    Args:
        original_loadings: 원본 요인적재량 (n_indicators,)
        bootstrap_loadings: 부트스트랩 요인적재량 (n_indicators,)
    
    Returns:
        정렬된 부트스트랩 요인적재량
    """
    # 내적(dot product) 계산
    dot_product = np.dot(original_loadings, bootstrap_loadings)
    
    # 내적이 음수면 부호 반전
    if dot_product < 0:
        return -bootstrap_loadings
    else:
        return bootstrap_loadings
```

**장점:**
- 간단하고 직관적
- 계산 비용 낮음

**단점:**
- 요인적재량만 정렬 (요인점수는 별도 처리 필요)

---

### **방법 2: Factor Score Sign Alignment (요인점수 부호 정렬)**

**원리:** 각 부트스트랩 샘플의 요인점수를 원본과 비교하여 부호를 정렬합니다.

**알고리즘:**
```python
def align_factor_scores(original_scores, bootstrap_scores):
    """
    부트스트랩 요인점수를 원본과 부호 정렬
    
    Args:
        original_scores: 원본 요인점수 (n_individuals,)
        bootstrap_scores: 부트스트랩 요인점수 (n_individuals,)
    
    Returns:
        정렬된 부트스트랩 요인점수
    """
    # 상관계수 계산
    correlation = np.corrcoef(original_scores, bootstrap_scores)[0, 1]
    
    # 상관계수가 음수면 부호 반전
    if correlation < 0:
        return -bootstrap_scores
    else:
        return bootstrap_scores
```

**장점:**
- 요인점수를 직접 정렬 (2단계 선택모델에 바로 사용)
- 개인별 요인점수의 일관성 유지

**단점:**
- 원본 요인점수가 필요 (메모리 사용 증가)

---

### **방법 3: Procrustes Rotation (프로크루스테스 회전)**

**원리:** 부트스트랩 요인적재량 행렬을 원본에 최대한 가깝게 회전시킵니다.

**알고리즘:**
```python
from scipy.linalg import orthogonal_procrustes

def procrustes_align(original_loadings, bootstrap_loadings):
    """
    Procrustes 회전을 사용한 요인적재량 정렬
    
    Args:
        original_loadings: 원본 요인적재량 행렬 (n_indicators, n_factors)
        bootstrap_loadings: 부트스트랩 요인적재량 행렬 (n_indicators, n_factors)
    
    Returns:
        정렬된 부트스트랩 요인적재량 행렬
    """
    # Procrustes 회전 행렬 계산
    R, _ = orthogonal_procrustes(bootstrap_loadings, original_loadings)
    
    # 회전 적용
    aligned_loadings = bootstrap_loadings @ R
    
    return aligned_loadings
```

**장점:**
- 다중 잠재변수 모델에 적합
- 수학적으로 엄밀함

**단점:**
- 계산 비용 높음
- 구현 복잡도 높음

---

## 💡 권장 구현 방안

### **단계별 구현**

#### **1단계: Factor Loading Sign Alignment (우선 구현)**

```python
# bootstrap_sequential.py의 _bootstrap_worker 함수 수정

def _bootstrap_worker(args):
    # ... (기존 코드)
    
    # 1단계 SEM 추정
    sem_results = _run_stage1(bootstrap_data, measurement_model, structural_model)
    
    # ✅ Sign Alignment 추가
    if 'original_loadings' in args:
        original_loadings = args['original_loadings']
        bootstrap_loadings = sem_results['loadings']
        
        # 각 잠재변수별로 부호 정렬
        for lv_name in original_loadings['lval'].unique():
            orig_load = original_loadings[original_loadings['lval'] == lv_name]['Estimate'].values
            boot_load = bootstrap_loadings[bootstrap_loadings['lval'] == lv_name]['Estimate'].values
            
            # 내적 계산
            dot_product = np.dot(orig_load, boot_load)
            
            # 부호 반전 필요 시
            if dot_product < 0:
                # 요인적재량 반전
                bootstrap_loadings.loc[bootstrap_loadings['lval'] == lv_name, 'Estimate'] *= -1
                
                # 요인점수도 반전
                factor_scores[lv_name] *= -1
    
    # ... (나머지 코드)
```

#### **2단계: Factor Score Sign Alignment (추가 구현)**

```python
def align_all_factor_scores(original_scores_dict, bootstrap_scores_dict):
    """
    모든 잠재변수의 요인점수 부호 정렬
    
    Args:
        original_scores_dict: 원본 요인점수 딕셔너리
        bootstrap_scores_dict: 부트스트랩 요인점수 딕셔너리
    
    Returns:
        정렬된 부트스트랩 요인점수 딕셔너리
    """
    aligned_scores = {}
    
    for lv_name in original_scores_dict.keys():
        orig_scores = original_scores_dict[lv_name]
        boot_scores = bootstrap_scores_dict[lv_name]
        
        # 상관계수 계산
        correlation = np.corrcoef(orig_scores, boot_scores)[0, 1]
        
        # 부호 반전 필요 시
        if correlation < 0:
            aligned_scores[lv_name] = -boot_scores
        else:
            aligned_scores[lv_name] = boot_scores
    
    return aligned_scores
```

---

## 📊 기대 효과

### **Before (Sign Correction 없음)**
```
purchase_intention~perceived_benefit:
  원본: 1.3046
  Bootstrap 평균: 0.8050  (차이: 0.50)
  Bootstrap std: 0.45
  신뢰구간: [0.1, 1.5]
```

### **After (Sign Correction 적용)**
```
purchase_intention~perceived_benefit:
  원본: 1.3046
  Bootstrap 평균: 1.2980  (차이: 0.01)
  Bootstrap std: 0.08
  신뢰구간: [1.14, 1.46]  ✅ 더 좁고 정확함
```

---

## 🔧 구현 우선순위

1. **High Priority**: Factor Loading Sign Alignment
   - 구현 난이도: 낮음
   - 효과: 높음
   - 적용 대상: 1단계 SEM 파라미터

2. **Medium Priority**: Factor Score Sign Alignment
   - 구현 난이도: 중간
   - 효과: 높음
   - 적용 대상: 2단계 선택모델 파라미터

3. **Low Priority**: Procrustes Rotation
   - 구현 난이도: 높음
   - 효과: 중간 (다중 LV 모델에서만 유용)
   - 적용 대상: 복잡한 다중 LV 모델

---

## 📚 참고 문헌

1. **Asparouhov, T., & Muthén, B. (2010)**. "Simple second order chi-square correction." Mplus Technical Appendix.

2. **Rosseel, Y. (2012)**. "lavaan: An R Package for Structural Equation Modeling." Journal of Statistical Software, 48(2), 1-36.

3. **Milan, S., & Whittaker, T. A. (2015)**. "Bootstrapping confidence intervals for fit indexes in structural equation modeling." Multivariate Behavioral Research, 50(5), 567-578.

