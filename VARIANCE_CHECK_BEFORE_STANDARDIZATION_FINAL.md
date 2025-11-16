# 요인점수 분산 체크 기능 최종 수정 완료 ✅

## 📋 수정 내용

### 문제점
이전 버전에서는 **Z-score 표준화 후**의 요인점수 분산을 체크하고 있었습니다.
- 표준화 후에는 모든 변수의 분산이 1.0이 되어버림
- 원본 요인점수의 분산이 작은지 판단할 수 없음

### 해결 방법
**표준화 전 원본 요인점수**의 분산을 체크하도록 수정했습니다.

## 🔧 수정된 코드

### 1. `estimate_stage1_only` 메서드 (Line 733-749)

```python
# 요인점수 추출 (SEM 결과에서)
original_factor_scores = sem_results['factor_scores']  # ✅ 원본 요인점수 보관
self.logger.info(f"요인점수 추출 완료: {list(original_factor_scores.keys())}")

# 요인점수 상세 로깅 (표준화 전)
self._log_factor_scores(original_factor_scores, stage="SEM 추출 직후 (표준화 전)")

# ✅ 표준화 전 분산 체크 (원본 요인점수)
self._check_factor_score_variance(original_factor_scores)

# 요인점수 Z-score 표준화
self.logger.info("\n요인점수 Z-score 표준화 적용...")
self.factor_scores = self._standardize_factor_scores(original_factor_scores)
self.logger.info("요인점수 표준화 완료")

# 표준화 후 로깅
self._log_factor_scores(self.factor_scores, stage="SEM 추출 직후 (표준화 후)")
```

### 2. 새로운 메서드: `_check_factor_score_variance` (Line 1155-1201)

```python
def _check_factor_score_variance(self, factor_scores: Dict[str, np.ndarray]) -> None:
    """
    요인점수 분산 체크 (표준화 전)

    각 잠재변수의 요인점수 분산을 계산하고, 분산이 너무 작은 경우 경고를 출력합니다.
    이 메서드는 표준화 이전의 원본 요인점수에 대해 호출되어야 합니다.
    """
    self.logger.info("\n" + "=" * 100)
    self.logger.info("요인점수 분산 체크 (표준화 전)")
    self.logger.info("=" * 100)
    
    low_variance_vars = []
    variance_threshold = 0.01  # 분산 임계값
    
    for lv_name, scores in factor_scores.items():
        variance = np.var(scores, ddof=0)  # 모집단 분산
        
        if variance < variance_threshold:
            low_variance_vars.append((lv_name, variance))
    
    if low_variance_vars:
        self.logger.warning("\n⚠️  분산이 너무 작은 요인점수 발견 (표준화 전):")
        for var_name, var_value in low_variance_vars:
            self.logger.warning(f"   - {var_name}: 분산 = {var_value:.6f}")
        self.logger.warning("   → 선택모델에서 비유의할 가능성이 높습니다.")
```

### 3. `_standardize_factor_scores` 메서드 간소화 (Line 1203-1261)

분산 체크 로직을 제거하고 표준화만 수행하도록 수정:

```python
def _standardize_factor_scores(self, factor_scores: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    요인점수 Z-score 표준화
    
    z = (x - mean(x)) / std(x)
    """
    standardized = {}
    
    for lv_name, scores in factor_scores.items():
        mean = np.mean(scores)
        std = np.std(scores, ddof=0)
        
        if std > 1e-10:
            standardized_scores = (scores - mean) / std
        else:
            standardized_scores = scores - mean
        
        standardized[lv_name] = standardized_scores
    
    return standardized
```

### 4. 결과 저장 시 원본 요인점수 포함 (Line 549-588)

```python
stage1_results = {
    'sem_results': sem_results,
    'factor_scores': self.factor_scores,  # 표준화된 요인점수
    'original_factor_scores': original_factor_scores,  # ✅ 원본 요인점수 (분산 체크용)
    'paths': sem_results['paths'],
    'loadings': sem_results['loadings'],
    ...
}
```

### 5. `save_stage1_results` 메서드 수정 (Line 1389-1444)

원본 요인점수의 통계를 저장하도록 수정:

```python
# 2-4. 요인점수 통계 저장 (원본 요인점수의 분산 포함)
if 'original_factor_scores' in results and results['original_factor_scores']:
    factor_stats_csv = f"{base_path}_factor_scores_stats.csv"
    stats_list = []
    variance_threshold = 0.01
    
    # ✅ 원본 요인점수의 통계 계산 (표준화 전)
    for lv_name, scores in results['original_factor_scores'].items():
        variance = np.var(scores, ddof=0)
        
        stats_list.append({
            'latent_variable': lv_name,
            'mean': np.mean(scores),
            'variance': variance,  # ✅ 원본 분산
            'std': np.std(scores, ddof=0),
            'min': np.min(scores),
            'max': np.max(scores),
            'n_observations': len(scores),
            'low_variance_warning': 'YES' if variance < variance_threshold else 'NO'
        })
```

## 📊 테스트 결과

### 테스트 데이터
- `health_consciousness`: 분산 = 0.002352 (매우 작음)
- `perceived_price`: 분산 = 0.006389 (작음)
- `purchase_intention`: 분산 = 1.346232 (정상)

### 저장된 통계 파일

```csv
latent_variable,mean,variance,std,min,max,n_observations,low_variance_warning
health_consciousness,5.5006,0.002352,0.0485,5.3379,5.6926,326,YES  ⚠️
perceived_price,3.7983,0.006389,0.0799,3.5842,4.0463,326,YES  ⚠️
purchase_intention,4.0886,1.346232,1.1603,0.8188,7.1589,326,NO  ✅
```

### 실제 데이터 결과

```csv
latent_variable,mean,variance,std,min,max,n_observations,low_variance_warning
health_concern,-0.000000,0.422245,0.649804,-2.5950,1.2884,326,NO  ✅
nutrition_knowledge,0.000000,0.327722,0.572470,-1.7764,1.2897,326,NO  ✅
perceived_benefit,-0.000000,0.351505,0.592879,-1.8105,1.4071,326,NO  ✅
perceived_price,0.000000,0.341933,0.584751,-1.7285,1.2535,326,NO  ✅
purchase_intention,0.000000,0.756235,0.869618,-2.4852,1.6194,326,NO  ✅
```

## ✅ 최종 확인 사항

1. **분산 체크 시점**: 표준화 **전** ✅
2. **저장되는 분산**: 원본 요인점수의 분산 ✅
3. **경고 플래그**: 분산 < 0.01인 경우 "YES" ✅
4. **하위 호환성**: `original_factor_scores`가 없으면 표준화된 요인점수 사용 ✅

## 📝 사용 방법

```python
# 1단계 추정 실행
results = estimator.estimate_stage1_only(
    data=data,
    measurement_model=measurement_model,
    structural_model=structural_model,
    save_path='results/stage1_results.pkl'
)

# 통계 파일 확인
stats_df = pd.read_csv('results/stage1_results_factor_scores_stats.csv')
low_var = stats_df[stats_df['low_variance_warning'] == 'YES']

if len(low_var) > 0:
    print("⚠️  분산이 작은 변수:")
    print(low_var[['latent_variable', 'variance']])
```

## 🎯 결론

이제 **표준화 전 원본 요인점수의 분산**을 정확히 체크하고 저장합니다!
- 분산이 작은 변수를 사전에 식별 가능
- 선택모델 투입 전 측정모델 개선 가능
- 비유의한 결과를 미리 예방 가능

---

**수정 완료일**: 2025-11-16  
**테스트 완료**: ✅  
**실제 데이터 검증**: ✅

