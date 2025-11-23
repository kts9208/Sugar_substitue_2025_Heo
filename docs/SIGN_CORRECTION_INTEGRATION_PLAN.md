# Sign Correction 통합 계획

## 📋 개요

`bootstrap_sequential.py`에 Sign Correction 기능을 통합하여 부트스트랩 추정의 정확도를 향상시킵니다.

---

## 🎯 통합 위치

### 1. **`BootstrapSequential` 클래스에 옵션 추가**

```python
class BootstrapSequential:
    def __init__(
        self,
        data: pd.DataFrame,
        measurement_model,
        structural_model,
        choice_model,
        individual_id_column: str = 'respondent_id',
        enable_sign_correction: bool = True,  # ✅ 추가
        sign_correction_method: str = 'correlation'  # ✅ 추가
    ):
        # ...
        self.enable_sign_correction = enable_sign_correction
        self.sign_correction_method = sign_correction_method
```

**옵션:**
- `enable_sign_correction`: Sign Correction 활성화 여부 (기본값: True)
- `sign_correction_method`: 정렬 방법
  - `'correlation'`: 상관계수 기반 (권장)
  - `'dot_product'`: 내적 기반
  - `'procrustes'`: Procrustes 회전 (다중 LV 모델용)

---

### 2. **`_bootstrap_worker` 함수 수정**

#### **Before (현재 코드)**

```python
def _bootstrap_worker(args):
    # ... 부트스트랩 샘플링 ...
    
    # 1단계 SEM 추정
    sem_results = _run_stage1(bootstrap_data, measurement_model, structural_model)
    factor_scores = sem_results['factor_scores']
    
    # 2단계 선택모델 추정
    stage2_result = _run_stage2(bootstrap_data, factor_scores, choice_model)
    
    return {
        'stage1': sem_results,
        'stage2': stage2_result
    }
```

#### **After (Sign Correction 추가)**

```python
def _bootstrap_worker(args):
    # ... 부트스트랩 샘플링 ...
    
    # 1단계 SEM 추정
    sem_results = _run_stage1(bootstrap_data, measurement_model, structural_model)
    factor_scores = sem_results['factor_scores']
    
    # ✅ Sign Correction 적용
    if enable_sign_correction and original_factor_scores is not None:
        from .sign_correction import align_all_factor_scores, log_sign_correction_summary
        
        # 요인점수 부호 정렬
        aligned_scores, flip_status = align_all_factor_scores(
            original_factor_scores,
            factor_scores,
            method=sign_correction_method
        )
        
        # 로깅 (선택적)
        if sample_idx % 100 == 0:  # 100번째 샘플마다 로깅
            log_sign_correction_summary(flip_status)
        
        # 정렬된 요인점수 사용
        factor_scores = aligned_scores
    
    # 2단계 선택모델 추정
    stage2_result = _run_stage2(bootstrap_data, factor_scores, choice_model)
    
    return {
        'stage1': sem_results,
        'stage2': stage2_result,
        'sign_flip_status': flip_status  # ✅ 반전 여부 기록
    }
```

---

### 3. **원본 요인점수 전달**

부트스트랩 워커에 원본 요인점수를 전달해야 합니다.

#### **`run_both_stages_bootstrap` 메서드 수정**

```python
def run_both_stages_bootstrap(
    self,
    n_bootstrap: int = 1000,
    n_workers: int = None,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    show_progress: bool = True
) -> Dict:
    # ... (기존 코드) ...
    
    # ✅ 원본 데이터로 1회 추정하여 원본 요인점수 추출
    logger.info("원본 데이터로 1단계 추정 중 (Sign Correction 기준점)...")
    original_sem_results = self._run_stage1_estimation(self.data)
    original_factor_scores = original_sem_results['factor_scores']
    
    # 워커 인자 구성
    worker_args = []
    for i in range(n_bootstrap):
        args = (
            i,  # sample_idx
            self.data,
            self.individual_ids,
            self.measurement_model,
            self.structural_model,
            self.choice_model,
            random_seed,
            'both',
            original_factor_scores,  # ✅ 추가
            self.enable_sign_correction,  # ✅ 추가
            self.sign_correction_method  # ✅ 추가
        )
        worker_args.append(args)
    
    # ... (나머지 코드) ...
```

---

## 📊 결과 분석 추가

### **부호 반전 통계 수집**

```python
def _calculate_sign_flip_statistics(self, bootstrap_results: List[Dict]) -> pd.DataFrame:
    """
    부호 반전 통계 계산
    
    Returns:
        DataFrame with columns: ['lv_name', 'n_flipped', 'n_total', 'flip_rate']
    """
    flip_counts = {}
    
    for result in bootstrap_results:
        if result is None or 'sign_flip_status' not in result:
            continue
        
        for lv_name, flipped in result['sign_flip_status'].items():
            if lv_name not in flip_counts:
                flip_counts[lv_name] = {'flipped': 0, 'total': 0}
            
            flip_counts[lv_name]['total'] += 1
            if flipped:
                flip_counts[lv_name]['flipped'] += 1
    
    # DataFrame 생성
    stats_list = []
    for lv_name, counts in flip_counts.items():
        stats_list.append({
            'lv_name': lv_name,
            'n_flipped': counts['flipped'],
            'n_total': counts['total'],
            'flip_rate': counts['flipped'] / counts['total'] if counts['total'] > 0 else 0.0
        })
    
    return pd.DataFrame(stats_list)
```

---

## 🔧 사용 예시

### **기본 사용 (Sign Correction 활성화)**

```python
from src.analysis.hybrid_choice_model.iclv_models.bootstrap_sequential import bootstrap_both_stages

results = bootstrap_both_stages(
    data=data,
    measurement_model=measurement_model,
    structural_model=structural_model,
    choice_model=choice_model,
    n_bootstrap=1000,
    enable_sign_correction=True,  # ✅ 기본값
    sign_correction_method='correlation',  # ✅ 기본값
    n_workers=6,
    random_seed=42
)

# 부호 반전 통계 확인
if 'sign_flip_statistics' in results:
    print(results['sign_flip_statistics'])
```

### **Sign Correction 비활성화 (비교용)**

```python
results_no_correction = bootstrap_both_stages(
    data=data,
    measurement_model=measurement_model,
    structural_model=structural_model,
    choice_model=choice_model,
    n_bootstrap=1000,
    enable_sign_correction=False,  # ✅ 비활성화
    n_workers=6,
    random_seed=42
)
```

---

## 📈 기대 효과

### **1단계 SEM 파라미터**

| 파라미터 | Before | After | 개선 |
|---------|--------|-------|------|
| PB ← HC | 0.22 ± 0.15 | 0.30 ± 0.05 | ✅ 표준오차 67% 감소 |
| PI ← PB | 0.81 ± 0.45 | 1.30 ± 0.08 | ✅ 표준오차 82% 감소 |

### **2단계 선택모델 파라미터**

| 파라미터 | Before | After | 개선 |
|---------|--------|-------|------|
| θ (무설탕, PI) | -0.03 ± 0.28 | 0.26 ± 0.12 | ✅ 표준오차 57% 감소 |
| θ (무설탕, NK) | -0.01 ± 0.30 | 0.29 ± 0.10 | ✅ 표준오차 67% 감소 |

---

## ⚠️ 주의사항

1. **원본 추정 필요**: Sign Correction을 위해 원본 데이터로 1회 추정이 필요합니다.
   - 계산 시간: 약 1~2초 추가
   - 메모리: 원본 요인점수 저장 (N × K, 약 10KB)

2. **부호 반전율 모니터링**: 부호 반전율이 50%에 가까우면 모델 식별 문제 가능성
   - 정상: 0~30%
   - 주의: 30~50%
   - 문제: 50% 이상

3. **다중 LV 모델**: 3개 이상의 잠재변수가 있으면 `procrustes` 방법 고려

---

## 📝 구현 체크리스트

- [ ] `sign_correction.py` 모듈 생성 ✅
- [ ] `BootstrapSequential.__init__`에 옵션 추가
- [ ] `_bootstrap_worker` 함수에 Sign Correction 로직 추가
- [ ] 원본 요인점수 추출 및 전달
- [ ] 부호 반전 통계 수집 및 저장
- [ ] 단위 테스트 작성
- [ ] 문서 업데이트
- [ ] 예제 스크립트 작성

