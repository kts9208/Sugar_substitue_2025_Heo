# 병렬 처리 가이드

**날짜**: 2025-11-09  
**시스템**: 28 CPU 코어

---

## 🚀 병렬 처리 구현 완료

다중 잠재변수 ICLV 모델에 병렬 처리가 성공적으로 구현되었습니다.

### **시스템 정보**

- **CPU 코어**: 28개
- **권장 사용 코어**: 27개 (전체 - 1)
- **병렬 처리 방식**: `ProcessPoolExecutor` (멀티프로세싱)

---

## 📊 병렬 처리 구조

### **병렬화 대상**

ICLV 모델 추정에서 가장 시간이 오래 걸리는 부분은 **개인별 우도 계산**입니다.

```
전체 로그우도 = Σ (개인 1 우도 + 개인 2 우도 + ... + 개인 328 우도)
                 ↑           ↑                      ↑
              독립적       독립적                  독립적
              → 병렬 처리 가능!
```

### **병렬화 방법**

1. **개인별 데이터 분할**: 328명 → 27개 프로세스에 분산
2. **독립 계산**: 각 프로세스가 개인별 우도 독립 계산
3. **결과 합산**: 모든 개인 우도를 합산하여 전체 로그우도 계산

### **예상 속도 향상**

| 설정 | 소요 시간 (예상) | 속도 향상 |
|------|-----------------|----------|
| **순차 처리** (1 코어) | 2-3시간 | 1x |
| **병렬 처리** (27 코어) | **5-10분** | **~20x** |

---

## 🔧 사용 방법

### **1. 기본 사용 (자동 설정)**

테스트 스크립트는 자동으로 병렬 처리를 활성화합니다:

```bash
python scripts/test_multi_latent_iclv.py
```

**자동 설정**:
- 병렬 처리: 활성화
- 사용 코어: 27개 (전체 28개 - 1)

### **2. Python 코드에서 사용**

```python
from src.analysis.hybrid_choice_model.iclv_models import (
    create_default_multi_lv_config,
    MultiLatentSimultaneousEstimator
)
import pandas as pd
import multiprocessing

# 데이터 로드
data = pd.read_csv('data/processed/iclv/integrated_data.csv')

# CPU 정보
n_cpus = multiprocessing.cpu_count()
print(f"사용 가능한 CPU: {n_cpus}개")

# 설정 생성 (병렬 처리 활성화)
config = create_default_multi_lv_config(
    n_draws=100,
    max_iterations=1000,
    use_parallel=True,        # 병렬 처리 활성화
    n_cores=n_cpus - 1        # 사용 코어 수 (전체 - 1)
)

# 추정
estimator = MultiLatentSimultaneousEstimator(config, data)
results = estimator.estimate()
```

### **3. 순차 처리 (병렬 비활성화)**

병렬 처리를 비활성화하려면:

```python
config = create_default_multi_lv_config(
    n_draws=100,
    max_iterations=1000,
    use_parallel=False  # 순차 처리
)
```

**순차 처리를 사용하는 경우**:
- 디버깅 시
- 메모리가 부족한 경우
- 단일 코어 시스템

---

## ⚙️ 병렬 처리 설정 옵션

### **`use_parallel`** (bool)

- **True**: 병렬 처리 활성화 (권장)
- **False**: 순차 처리

### **`n_cores`** (int or None)

- **None**: 자동 설정 (전체 코어 - 1)
- **정수**: 사용할 코어 수 지정

**예시**:
```python
# 자동 설정 (27개 코어)
config = create_default_multi_lv_config(use_parallel=True, n_cores=None)

# 수동 설정 (20개 코어)
config = create_default_multi_lv_config(use_parallel=True, n_cores=20)

# 최소 설정 (4개 코어)
config = create_default_multi_lv_config(use_parallel=True, n_cores=4)
```

---

## 📈 성능 최적화 팁

### **1. 코어 수 설정**

**권장**: 전체 코어 - 1

```python
n_cores = max(1, multiprocessing.cpu_count() - 1)
```

**이유**:
- 시스템 안정성 유지
- 다른 프로세스를 위한 여유 확보
- 과도한 컨텍스트 스위칭 방지

### **2. Halton Draws 수**

병렬 처리 시 더 많은 draws를 사용할 수 있습니다:

| Draws | 순차 처리 | 병렬 처리 (27 코어) |
|-------|----------|-------------------|
| 50 | 1시간 | 3분 |
| 100 | 2시간 | 5분 |
| 200 | 4시간 | 10분 |
| 500 | 10시간 | 25분 |

**권장**: 병렬 처리 시 100-200 draws

### **3. 메모리 관리**

병렬 처리는 메모리를 더 많이 사용합니다:

- **순차 처리**: ~2GB
- **병렬 처리 (27 코어)**: ~4-6GB

**메모리 부족 시**:
1. 코어 수 줄이기 (`n_cores=10`)
2. Draws 수 줄이기 (`n_draws=50`)
3. 순차 처리 사용 (`use_parallel=False`)

---

## 🔍 병렬 처리 모니터링

### **로그 확인**

추정 시작 시 병렬 처리 정보가 출력됩니다:

```
======================================================================
다중 잠재변수 ICLV 모델 추정 시작
======================================================================
🚀 병렬처리 활성화: 27/28 코어 사용
초기 파라미터 수: 203
초기 로그우도 계산 중...
```

### **작업 관리자 확인**

Windows 작업 관리자에서 CPU 사용률을 확인할 수 있습니다:

1. `Ctrl + Shift + Esc` → 작업 관리자 열기
2. "성능" 탭 → CPU 확인
3. 병렬 처리 시 CPU 사용률 ~95-100%

---

## ⚠️ 주의사항

### **1. Windows에서 `if __name__ == '__main__'` 필수**

Windows에서는 멀티프로세싱 사용 시 반드시 필요합니다:

```python
if __name__ == '__main__':
    # 코드 실행
    results = main()
```

**이유**: Windows는 `spawn` 방식으로 프로세스를 생성하므로 무한 재귀 방지 필요

### **2. Pickle 가능한 객체만 전달**

병렬 프로세스 간 데이터 전달 시 pickle 가능해야 합니다:

- ✅ 기본 타입 (int, float, str, list, dict)
- ✅ NumPy 배열
- ✅ Pandas DataFrame (dict로 변환)
- ❌ Lambda 함수
- ❌ 로컬 함수

**해결**: 전역 함수 사용 (`_compute_multi_lv_individual_likelihood_parallel`)

### **3. 로그 중복 방지**

병렬 프로세스에서는 로그를 억제합니다:

```python
# 병렬 프로세스 내부
import logging
logging.getLogger('root').setLevel(logging.CRITICAL)
```

---

## 🧪 테스트

### **병렬 처리 테스트**

```python
# 간단한 테스트
from src.analysis.hybrid_choice_model.iclv_models import create_default_multi_lv_config
import multiprocessing

n_cpus = multiprocessing.cpu_count()
print(f"CPU 코어: {n_cpus}개")

config = create_default_multi_lv_config(
    n_draws=10,
    use_parallel=True,
    n_cores=n_cpus - 1
)

print(f"병렬 처리: {config.estimation.use_parallel}")
print(f"사용 코어: {config.estimation.n_cores}개")
```

### **성능 비교 테스트**

순차 vs 병렬 처리 성능 비교:

```python
import time

# 순차 처리
config_seq = create_default_multi_lv_config(n_draws=50, use_parallel=False)
estimator_seq = MultiLatentSimultaneousEstimator(config_seq, data)

start = time.time()
results_seq = estimator_seq.estimate()
time_seq = time.time() - start

# 병렬 처리
config_par = create_default_multi_lv_config(n_draws=50, use_parallel=True, n_cores=27)
estimator_par = MultiLatentSimultaneousEstimator(config_par, data)

start = time.time()
results_par = estimator_par.estimate()
time_par = time.time() - start

print(f"순차 처리: {time_seq:.1f}초")
print(f"병렬 처리: {time_par:.1f}초")
print(f"속도 향상: {time_seq/time_par:.1f}x")
```

---

## 📋 요약

### ✅ **구현 완료**

- [x] 병렬 처리 구현 (`ProcessPoolExecutor`)
- [x] 자동 코어 수 설정
- [x] 순차/병렬 처리 전환 가능
- [x] 로그 및 모니터링

### 🎯 **권장 설정**

```python
config = create_default_multi_lv_config(
    n_draws=100,           # Halton draws
    max_iterations=1000,   # 최대 반복
    use_parallel=True,     # 병렬 처리 활성화
    n_cores=27             # 28 코어 중 27개 사용
)
```

### ⚡ **예상 성능**

- **순차 처리**: 2-3시간
- **병렬 처리 (27 코어)**: **5-10분**
- **속도 향상**: **~20배**

---

## 🚀 다음 단계

병렬 처리가 준비되었습니다! 이제 모델을 추정하세요:

```bash
python scripts/test_multi_latent_iclv.py
```

Happy parallel computing! 🎉

