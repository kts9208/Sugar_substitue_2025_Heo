# 동시추정 계산속도 최적화 제안

## 📊 현재 성능 분석 (2025-11-23 실행 기준)

### **성능 지표**
- **총 소요 시간**: 45.4분 (2,726초)
- **Iteration 수**: 11회
- **Iteration당 평균 시간**: 238.2초 (약 4분)
- **최대 Iteration 시간**: 363초 (6분)
- **최소 Iteration 시간**: 192초 (3.2분)

### **문제점**
1. ✗ **Iteration당 평균 4분 소요** - 매우 느림
2. ✗ **10번의 iteration에서 LL 악화** - 수렴 불안정
3. ✗ **전체 추정에 45분 소요** - 실용성 낮음

---

## 💡 최적화 제안 (우선순위별)

### **[HIGH] 제안 1: Halton Draws 수 감소**

#### **현재 설정**
```python
N_DRAWS = 100  # scripts/test_gpu_batch_iclv.py
```

#### **문제**
- 100 draws는 정확도는 높지만 계산 비용이 매우 큼
- 328명 × 100 draws = 32,800개의 우도 계산 필요
- Iteration당 4분 소요의 주요 원인

#### **제안**
```python
# 단계별 테스트 권장
N_DRAWS = 50   # 1단계: 50% 속도 향상 예상
N_DRAWS = 30   # 2단계: 70% 속도 향상 예상 (권장)
N_DRAWS = 20   # 3단계: 80% 속도 향상 예상 (최소 권장값)
```

#### **예상 효과**
| Draws | 예상 시간/Iter | 예상 총 시간 | 정확도 손실 |
|-------|---------------|-------------|-----------|
| 100 (현재) | 238초 | 45분 | 기준 |
| 50 | 119초 | 22분 | 매우 낮음 |
| 30 | 71초 | 13분 | 낮음 |
| 20 | 48초 | 9분 | 중간 |

#### **구현**
```python
# scripts/test_gpu_batch_iclv.py 수정
N_DRAWS = 30  # 100 → 30으로 변경
```

---

### **[HIGH] 제안 2: 로깅 레벨 최소화**

#### **현재 설정**
```python
# src/analysis/hybrid_choice_model/iclv_models/iclv_config.py
gradient_log_level: Literal['MINIMAL', 'MODERATE', 'DETAILED'] = 'DETAILED'
```

#### **문제**
- DETAILED 로깅은 모든 중간 계산 과정을 기록
- 파일 I/O 오버헤드 발생
- 로그 파일 크기 증가 (84KB)

#### **제안**
```python
gradient_log_level = 'MINIMAL'  # DETAILED → MINIMAL
```

#### **예상 효과**
- Iteration당 5-10초 절약 예상
- 로그 파일 크기 90% 감소

---

### **[MEDIUM] 제안 3: Line Search 파라미터 조정**

#### **현재 상황**
- Line Search 호출 분포:
  - 1회: 8번 (73%)
  - 2회: 3번 (27%)
- 10번의 iteration에서 LL 악화 발생

#### **문제**
- Line Search가 Wolfe 조건을 만족하지 못함
- 수렴이 불안정함

#### **제안**
```python
# scipy.optimize.minimize의 L-BFGS-B 옵션 조정
options = {
    'maxls': 20,      # 현재 10 → 20으로 증가
    'ftol': 1e-4,     # 현재 1e-3 → 완화
    'gtol': 1e-4,     # 현재 1e-3 → 완화
}
```

#### **예상 효과**
- 수렴 안정성 향상
- LL 악화 빈도 감소

---

### **[MEDIUM] 제안 4: GPU 메모리 임계값 증가**

#### **현재 설정**
```python
CPU_MEMORY_THRESHOLD_MB = 2000  # 2GB
GPU_MEMORY_THRESHOLD_MB = 5000  # 5GB
```

#### **제안**
```python
# GPU 메모리가 충분하다면 (8GB 이상)
GPU_MEMORY_THRESHOLD_MB = 7000  # 5GB → 7GB
```

#### **예상 효과**
- GPU 배치 크기 증가 가능
- 메모리 정리 빈도 감소
- Iteration당 2-5초 절약 예상

---

### **[LOW] 제안 5: 초기값 개선**

#### **현재 설정**
```python
# 모든 파라미터를 0.1로 초기화
initial_params = {
    'structural': {'gamma_...': 0.1, ...},
    'choice': {'asc_...': 0.1, ...}
}
```

#### **제안**
순차추정 2단계 결과를 더 정확하게 사용:
```python
# st2_HC-PB_PB-PI1_PI2_results.csv에서 추정값 로드
initial_params = load_from_sequential_stage2(csv_path)
```

#### **예상 효과**
- 초기 iteration 수 감소 (11회 → 7-8회 예상)
- 총 시간 20-30% 절약

---

## 🎯 권장 최적화 조합

### **조합 1: 빠른 테스트용 (권장)**
```python
N_DRAWS = 30                      # 100 → 30
gradient_log_level = 'MINIMAL'    # DETAILED → MINIMAL
```
**예상 효과**: 45분 → **10-12분** (75% 단축)

### **조합 2: 균형잡힌 설정**
```python
N_DRAWS = 50                      # 100 → 50
gradient_log_level = 'MINIMAL'    # DETAILED → MINIMAL
GPU_MEMORY_THRESHOLD_MB = 7000    # 5GB → 7GB
```
**예상 효과**: 45분 → **18-20분** (55% 단축)

### **조합 3: 최대 정확도 유지**
```python
N_DRAWS = 100                     # 유지
gradient_log_level = 'MINIMAL'    # DETAILED → MINIMAL
GPU_MEMORY_THRESHOLD_MB = 7000    # 5GB → 7GB
options = {'maxls': 20, 'ftol': 1e-4, 'gtol': 1e-4}
```
**예상 효과**: 45분 → **35-38분** (20% 단축)

---

## 📝 구현 가이드

### **1단계: Halton Draws 감소 (가장 효과적)**

**파일**: `scripts/test_gpu_batch_iclv.py`

```python
# Line 124 수정
N_DRAWS = 30  # 100 → 30으로 변경
```

### **2단계: 로깅 레벨 최소화**

**파일**: `scripts/test_gpu_batch_iclv.py`

```python
# Config 생성 시 추가
config = create_sugar_substitute_multi_lv_config(
    custom_paths=hierarchical_paths,
    choice_config_overrides=choice_config_dict,
    n_draws=N_DRAWS,
    max_iterations=MAX_ITERATIONS,
    optimizer='L-BFGS-B',
    use_analytic_gradient=True,
    gradient_log_level='MINIMAL',  # 추가
)
```

### **3단계: 테스트 실행**

```bash
python scripts/test_gpu_batch_iclv.py
```

### **4단계: 결과 비교**

```bash
python analyze_computation_speed.py
```

---

## ⚠️ 주의사항

1. **Draws 수 감소 시**:
   - 30 draws 이하로 내리면 정확도 손실 가능
   - 최종 결과는 100 draws로 재실행 권장

2. **로깅 레벨 변경 시**:
   - 디버깅이 어려워질 수 있음
   - 문제 발생 시 DETAILED로 복원

3. **GPU 메모리 증가 시**:
   - GPU 메모리 부족 에러 발생 가능
   - 시스템 GPU 메모리 확인 필요

---

## 📊 예상 성능 개선 요약

| 최적화 항목 | 예상 시간 절약 | 구현 난이도 | 정확도 영향 |
|------------|--------------|-----------|----------|
| Draws 30으로 감소 | 30-35분 | 매우 쉬움 | 낮음 |
| 로깅 MINIMAL | 1-2분 | 매우 쉬움 | 없음 |
| GPU 메모리 증가 | 0.5-1분 | 쉬움 | 없음 |
| Line Search 조정 | 2-5분 | 중간 | 없음 |
| 초기값 개선 | 5-10분 | 중간 | 없음 |

**총 예상 절약**: **38-53분** → **최종 시간: 5-15분**

---

## ✅ 다음 단계

1. **조합 1 테스트** (N_DRAWS=30, MINIMAL 로깅)
2. **결과 검증** (LL, 파라미터 추정값 비교)
3. **정확도 확인** (100 draws 결과와 비교)
4. **최종 설정 결정**

