# GPU Analytic Gradient 문제점 요약

## 🔍 자동 진단 결과

```bash
$ python scripts/diagnose_gpu_gradient.py

발견된 문제: 5개
  🔴 CRITICAL: 3개
  🟡 MAJOR: 2개

결론: GPU Analytic Gradient는 현재 사용 불가능합니다.
```

---

## 🔴 CRITICAL 문제 (3개)

### 1. Importance Weighting 누락

**문제:**
```python
# 현재 GPU 구현 (잘못됨)
grad_zeta = cp.asnumpy(grad_zeta_batch.sum(axis=0))  # ❌ 단순 합산
```

**올바른 구현:**
```python
# CPU 구현 (올바름)
weights = np.array(draw_likelihoods) / total_likelihood
for w, grad in zip(weights, draw_gradients):
    weighted_grad += w * grad  # ✅ 가중평균
```

**영향:**
- 수학적 오류: 시뮬레이션 기반 추정 원리 위반
- NaN 발생: 극단적인 그래디언트 누적
- 수렴 실패: 잘못된 방향으로 이동

---

### 2. 측정모델 - 첫 번째 행만 사용

**문제:**
```python
# gpu_gradient_batch.py, Line 74
first_row = ind_data.iloc[0]  # ❌ 첫 번째 행만

for i, indicator in enumerate(config.indicators):
    y = first_row[indicator]  # ❌ 첫 번째 행의 값만
```

**영향:**
```
개인 데이터: 18개 선택 상황
  Row 0: ✅ 사용됨
  Row 1-17: ❌ 무시됨

→ 94.4% 데이터 손실!
```

**올바른 구현:**
```python
# 모든 행 순회
for idx in range(len(ind_data)):
    row = ind_data.iloc[idx]
    for i, indicator in enumerate(config.indicators):
        y = row[indicator]  # ✅ 각 행의 값
```

---

### 3. Likelihood 계산 함수 누락

**문제:**
- GPU 파일에 likelihood 계산 함수 없음
- Importance weighting을 위한 가중치 계산 불가능

**필요한 구현:**
```python
def compute_likelihood_batch_gpu(ind_data, lvs_list, params):
    """각 draw의 likelihood 계산"""
    ll_batch = cp.zeros(n_draws)
    
    for draw_idx in range(n_draws):
        ll_meas = compute_measurement_ll_gpu(...)
        ll_struct = compute_structural_ll_gpu(...)
        ll_choice = compute_choice_ll_gpu(...)
        ll_batch[draw_idx] = ll_meas + ll_struct + ll_choice
    
    return ll_batch
```

---

## 🟡 MAJOR 문제 (2개)

### 4. NaN 체크 누락

**현재 상태:**
- Clipping은 일부 구현됨 (prob)
- NaN 체크 코드 없음
- Log-sum-exp trick 없음

**필요한 개선:**
```python
# NaN 체크
if cp.any(cp.isnan(grad_zeta)):
    logger.warning("NaN detected")
    grad_zeta = cp.nan_to_num(grad_zeta, nan=0.0)

# Log-sum-exp trick
def log_sum_exp(log_values):
    max_val = cp.max(log_values)
    return max_val + cp.log(cp.sum(cp.exp(log_values - max_val)))

# Gradient clipping
grad_zeta = cp.clip(grad_zeta, -1e6, 1e6)
```

---

### 5. 선택모델 순차 처리

**문제:**
```python
# gpu_gradient_batch.py, Line 275
for draw_idx in range(n_draws):  # ❌ 순차 처리
    lv = lv_gpu[draw_idx]
    V = intercept + cp.dot(attr_gpu, beta_gpu) + lambda_lv * lv
    # ...
```

**개선:**
```python
# 배치 처리
lv_batch = lv_gpu[:, None]  # (n_draws, 1)
V_batch = intercept + cp.dot(attr_gpu, beta_gpu) + lambda_lv * lv_batch
# Shape: (n_draws, n_situations)
# → GPU 병렬 처리
```

---

## 📊 CPU vs GPU 비교

| 기능 | CPU 구현 | GPU 구현 | 상태 |
|------|---------|---------|------|
| **Importance weighting** | ✅ 구현됨 | ❌ 누락 | CRITICAL |
| **측정모델 모든 행** | ✅ 구현됨 | ❌ 첫 행만 | CRITICAL |
| **Likelihood 계산** | ✅ 구현됨 | ❌ 누락 | CRITICAL |
| **가중평균** | ✅ 구현됨 | ❌ 단순 합산 | CRITICAL |
| **NaN 체크** | ✅ 구현됨 | ❌ 누락 | MAJOR |
| **배치 처리** | ❌ 순차 | ❌ 순차 | MAJOR |

---

## 🔧 수정 계획

### Phase 1: Critical 문제 수정 (4-6시간)

1. **Likelihood 계산 함수 추가** (2시간)
   - `compute_measurement_ll_gpu()`
   - `compute_structural_ll_gpu()`
   - `compute_choice_ll_gpu()`
   - `compute_likelihood_batch_gpu()`

2. **Importance weighting 구현** (1시간)
   ```python
   ll_batch = compute_likelihood_batch_gpu(...)
   weights = cp.exp(ll_batch) / cp.sum(cp.exp(ll_batch))
   grad_weighted = cp.sum(weights[:, None] * grad_batch, axis=0)
   ```

3. **측정모델 모든 행 처리** (1시간)
   ```python
   for idx in range(len(ind_data)):
       row = ind_data.iloc[idx]
       # 그래디언트 계산
   ```

4. **단순 합산 → 가중평균** (1시간)
   - 모든 `.sum(axis=0)`를 가중평균으로 변경

### Phase 2: Major 문제 수정 (2-3시간)

5. **수치 안정성 강화** (1-2시간)
   - Log-sum-exp trick
   - NaN 체크 및 처리
   - Gradient clipping

6. **선택모델 배치 처리** (1시간)
   - For loop 제거
   - Broadcasting 사용

### Phase 3: 테스트 및 검증 (2-3시간)

7. **단위 테스트 작성**
   - 각 함수별 테스트
   - CPU vs GPU 결과 비교

8. **통합 테스트**
   - 전체 그래디언트 계산
   - Numerical gradient와 비교

---

## 📈 예상 결과

### 수정 전 (현재)

```
상태: ❌ 사용 불가능
이유: NaN 에러, 수학적 오류
시간: N/A (실패)
```

### 수정 후

```
상태: ✅ 사용 가능
정확도: ✅ 올바름 (CPU와 동일)
시간: ~22초/그래디언트
속도 향상: 77분 → 22초 (210배)
```

---

## 💡 권장사항

### 단기 (현재)

**Numerical gradient 사용**
- ✅ 안정적이고 검증됨
- ✅ GPU 우도 계산 활용
- ⏱️ 77분/그래디언트 (느리지만 작동)

### 중기 (1-2주)

**GPU gradient 수정**
- 🔧 Critical 문제 수정 (4-6시간)
- 🔧 Major 문제 수정 (2-3시간)
- ✅ 테스트 및 검증 (2-3시간)
- 🎯 총 예상 시간: **8-12시간**

### 장기 (1개월+)

**하이브리드 접근**
- 초기 iterations: Numerical (안정적)
- 후기 iterations: GPU (빠른 수렴)
- 자동 전환 로직

---

## 📚 관련 문서

1. **`docs/gpu_gradient_problems_analysis.md`**
   - 전체 문제점 상세 분석 (300줄)
   - 각 문제별 코드 비교
   - 수정 방법 제시

2. **`scripts/diagnose_gpu_gradient.py`**
   - 자동 진단 스크립트
   - 문제점 자동 검출
   - Exit code로 심각도 반환

3. **`docs/gradient_comparison_summary.md`**
   - Numerical vs Analytic 비교
   - 계산량 분석
   - 성능 비교

---

## 🎯 결론

**현재 GPU Analytic Gradient는 사용 불가능합니다.**

**주요 이유:**
1. 🔴 Importance weighting 누락 → 수학적 오류
2. 🔴 94% 데이터 무시 → 정보 손실
3. 🔴 Likelihood 계산 불가 → Weighting 불가능

**수정 가능성:**
- ✅ 모든 문제 수정 가능
- ⏱️ 예상 시간: 8-12시간
- 🚀 수정 후 이득: 210배 속도 향상

**현재 최선의 선택:**
- Numerical gradient 사용 (77분, 안정적)
- GPU gradient 수정은 중기 과제로 설정

