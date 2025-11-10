# 메모리 최적화 가이드

## 📊 개요

ICLV 모델 추정 시 메모리 과부하를 방지하고 안정적인 실행을 보장하기 위한 메모리 관리 방안입니다.

---

## 🔍 강제종료 원인 분석

### **발생 상황**
- **시간**: 2025-11-10 09:50:49
- **Iteration**: 45/200 (22.5% 완료)
- **실행 시간**: 3시간 47분
- **종료 시점**: Gradient 계산 중

### **주요 원인**

#### 1. **장시간 실행** ⚠️
- 예상 총 소요 시간: 16-17시간
- 조기 종료 비활성화 (`patience=999999`)
- 최대 반복 200회 설정

#### 2. **메모리 누적** ⚠️
- 326명 × 100 draws × 202 파라미터
- 각 iteration마다 임시 배열 생성
- 가비지 컬렉션 미실행

#### 3. **GPU/CPU 메모리 사용 패턴**
```
GPU 메모리: 2,014 MB / 8,188 MB (24.6%) ✅ 여유 있음
CPU 메모리: 확인 불가 ❌ 부족 가능성
```

---

## ✅ 해결 방안

### **방안 1: 메모리 모니터링 및 자동 정리** (구현 완료)

#### **MemoryMonitor 클래스**

**위치**: `src/analysis/hybrid_choice_model/iclv_models/memory_monitor.py`

**기능**:
1. CPU/GPU 메모리 사용량 실시간 모니터링
2. 임계값 초과 시 자동 가비지 컬렉션
3. 메모리 사용 기록 및 통계

**사용법**:
```python
from .memory_monitor import MemoryMonitor

# 초기화
memory_monitor = MemoryMonitor(
    cpu_threshold_mb=2000,  # CPU 메모리 임계값 (MB)
    gpu_threshold_mb=1500,  # GPU 메모리 임계값 (MB)
    auto_cleanup=True       # 자동 정리 활성화
)

# 메모리 체크 및 정리
mem_info = memory_monitor.check_and_cleanup("우도 계산")

# 메모리 통계 로깅
memory_monitor.log_memory_stats("Iteration 10")
```

#### **GPUBatchEstimator 통합**

**변경 사항**:
```python
class GPUBatchEstimator(SimultaneousEstimator):
    def __init__(self, config, use_gpu=True,
                 memory_monitor_cpu_threshold_mb=2000,
                 memory_monitor_gpu_threshold_mb=1500):
        # 메모리 모니터 초기화
        self.memory_monitor = MemoryMonitor(
            cpu_threshold_mb=memory_monitor_cpu_threshold_mb,
            gpu_threshold_mb=memory_monitor_gpu_threshold_mb,
            auto_cleanup=True
        )
```

**적용 위치**:
1. **우도 계산 전**: 개인별 우도 계산 시작 시
2. **측정모델 계산 후**: `gc.collect()` 호출
3. **선택모델 계산 후**: `gc.collect()` 호출
4. **구조모델 계산 후**: `gc.collect()` 호출

---

### **방안 2: 배치 크기 조정** (선택적)

현재 전체 326명을 한 번에 처리하는 대신, 배치로 나누어 처리:

```python
# 예시: 100명씩 배치 처리
batch_size = 100
n_batches = (n_individuals + batch_size - 1) // batch_size

for batch_idx in range(n_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, n_individuals)
    
    # 배치 처리
    batch_ll = process_batch(start_idx, end_idx)
    
    # 메모리 정리
    gc.collect()
```

**효과**:
- CPU 메모리 사용량 감소
- 계산 안정성 향상
- 약간의 속도 저하 (배치 간 오버헤드)

---

### **방안 3: 명시적 배열 삭제**

임시 배열을 명시적으로 삭제:

```python
from .memory_monitor import cleanup_arrays

# 계산 수행
ll_measurement_batch = compute_measurement_batch_gpu(...)
ll_choice_batch = compute_choice_batch_gpu(...)

# 사용 후 삭제
cleanup_arrays(ll_measurement_batch, ll_choice_batch)
```

---

## 📈 메모리 사용량 추정

### **우도 계산 시**

**배열 크기**:
- `lvs_list`: 100 draws × 5 LVs × 8 bytes = 4 KB
- `ll_measurement_batch`: 100 draws × 8 bytes = 0.8 KB
- `ll_choice_batch`: 100 draws × 8 bytes = 0.8 KB
- `ll_structural_batch`: 100 draws × 8 bytes = 0.8 KB

**총 메모리 (개인당)**: ~6 KB
**전체 (326명)**: ~2 MB

### **그래디언트 계산 시**

**배열 크기**:
- `grad_zeta_batch`: 100 draws × 38 indicators × 8 bytes = 30 KB
- `grad_tau_batch`: 100 draws × 38 × 4 × 8 bytes = 122 KB
- `grad_gamma_lv`: 4 × 8 bytes = 32 bytes
- `grad_gamma_x`: 3 × 8 bytes = 24 bytes

**총 메모리 (개인당)**: ~152 KB
**전체 (326명)**: ~50 MB

### **누적 메모리**

**45 iterations 후**:
- 우도 계산: 45 × 2 MB = 90 MB
- 그래디언트 계산: 45 × 50 MB = 2,250 MB ⚠️

**문제점**: 가비지 컬렉션 없이 누적 시 **2.3 GB** 사용

---

## 🎯 권장 설정

### **메모리 임계값**

```python
# 보수적 설정 (안정성 우선)
memory_monitor_cpu_threshold_mb=1500  # 1.5 GB
memory_monitor_gpu_threshold_mb=1000  # 1 GB

# 표준 설정 (균형)
memory_monitor_cpu_threshold_mb=2000  # 2 GB
memory_monitor_gpu_threshold_mb=1500  # 1.5 GB

# 공격적 설정 (성능 우선)
memory_monitor_cpu_threshold_mb=3000  # 3 GB
memory_monitor_gpu_threshold_mb=2000  # 2 GB
```

### **가비지 컬렉션 빈도**

```python
# 매 iteration마다 (안정성 최대)
gc.collect()  # 우도 계산 후
gc.collect()  # 그래디언트 계산 후

# 5 iterations마다 (균형)
if iteration % 5 == 0:
    gc.collect()

# 10 iterations마다 (성능 우선)
if iteration % 10 == 0:
    gc.collect()
```

---

## 📊 모니터링 로그 예시

### **정상 상태**
```
[메모리 상태] Iteration 10
  프로세스 CPU: 1,234.5MB
  시스템 전체: 8,456.7MB / 16,384.0MB (51.6%)
  시스템 여유: 7,927.3MB
  GPU: 1,234.5MB
```

### **임계값 초과**
```
[메모리 경고] 우도 계산 - 개인 123 | CPU: 2,345.6MB (임계값: 2000MB)
[메모리 정리 완료] CPU: 345.6MB 해제, GPU: 123.4MB 해제
```

---

## 🔧 테스트 방법

### **1. 메모리 모니터 단독 테스트**

```python
from src.analysis.hybrid_choice_model.iclv_models.memory_monitor import MemoryMonitor

monitor = MemoryMonitor(
    cpu_threshold_mb=1000,
    gpu_threshold_mb=500,
    auto_cleanup=True
)

# 메모리 사용량 확인
monitor.log_memory_stats("테스트 시작")

# 대용량 배열 생성
import numpy as np
large_array = np.random.rand(10000, 10000)  # ~800 MB

# 메모리 체크
mem_info = monitor.check_and_cleanup("대용량 배열 생성 후")

# 배열 삭제
del large_array

# 정리 후 확인
monitor.log_memory_stats("정리 후")
```

### **2. GPU Batch Estimator 테스트**

```python
# test_gpu_batch_iclv.py에서
estimator = GPUBatchEstimator(
    config,
    use_gpu=True,
    memory_monitor_cpu_threshold_mb=1500,  # 낮은 임계값으로 테스트
    memory_monitor_gpu_threshold_mb=1000
)

# 추정 실행
results = estimator.estimate(...)

# 메모리 요약 확인
summary = estimator.memory_monitor.get_memory_summary()
print(f"최대 CPU 메모리: {summary['cpu_max_mb']:.1f}MB")
print(f"평균 CPU 메모리: {summary['cpu_avg_mb']:.1f}MB")
```

---

## 📝 요약

| 방안 | 구현 상태 | 효과 | 성능 영향 |
|------|----------|------|----------|
| **메모리 모니터링** | ✅ 완료 | 과부하 방지 | 최소 (~1%) |
| **자동 가비지 컬렉션** | ✅ 완료 | 메모리 누적 방지 | 작음 (~2-3%) |
| **배치 크기 조정** | ⏸️ 선택적 | 메모리 사용량 감소 | 중간 (~5-10%) |
| **명시적 배열 삭제** | ⏸️ 선택적 | 메모리 즉시 해제 | 최소 (~1%) |

**권장 조합**: 메모리 모니터링 + 자동 가비지 컬렉션 (현재 구현)

