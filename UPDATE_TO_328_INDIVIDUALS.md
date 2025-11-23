# 328명 데이터로 업데이트 가이드

**작성 날짜:** 2025-11-23  
**작성자:** ICLV Team

---

## 📋 현황 요약

### ✅ **현재 상태**

| 항목 | 개인 수 | 상태 |
|------|---------|------|
| **통합 데이터셋** (`integrated_data.csv`) | **328명** | ✅ 최신 |
| **동시추정 결과** | **328명** | ✅ 최신 |
| **1단계 순차추정 결과** | **326명** | ❌ 구버전 |
| **CFA 결과** | **326명** | ❌ 구버전 |
| **백업 파일들** | **326명** | ❌ 구버전 |

### ⚠️ **문제점**

1. **통합 데이터셋은 328명**으로 업데이트되었지만
2. **CFA 결과와 1단계 순차추정 결과는 326명** 데이터로 추정됨
3. **2단계 순차추정 실행 시 개인 수 불일치 발생 가능**

---

## 🎯 재실행 필요 파일

### 1️⃣ **CFA Only (측정모델만)**

**파일:** `examples/sequential_cfa_only_example.py`

**실행 방법:**
```bash
python examples/sequential_cfa_only_example.py
```

**생성 파일:**
```
results/sequential_stage_wise/
├── cfa_results.pkl                              # 전체 결과
├── cfa_results_factor_scores.csv                # 요인점수
├── cfa_results_loadings.csv                     # 요인적재량
├── cfa_results_fit_indices.csv                  # 적합도
├── cfa_results_all_params.csv                   # 모든 파라미터
├── cfa_results_measurement_params.csv           # 측정모델 파라미터
├── cfa_results_correlation_matrix.csv           # 상관행렬
└── cfa_results_pvalue_matrix.csv                # p-value 행렬
```

**예상 시간:** 약 1-2분

---

### 2️⃣ **1단계 순차추정 (SEM: 측정모델 + 구조모델)**

**파일:** `examples/sequential_stage1.py`

**현재 설정 확인:**
```python
PATHS = {
    'HC->PB': True,   # 건강관심도 → 건강유익성
    'HC->PP': False,  # 건강관심도 → 가격수준
    'HC->PI': False,  # 건강관심도 → 구매의도
    'PB->PI': True,   # 건강유익성 → 구매의도
    'PP->PI': False,  # 가격수준 → 구매의도
    'NK->PI': False,  # 영양지식 → 구매의도
}
STANDARDIZATION_METHOD = 'zscore'  # Z-score 표준화
```

**실행 방법:**
```bash
python examples/sequential_stage1.py
```

**생성 파일:**
```
results/sequential_stage_wise/
├── stage1_HC-PB_PB-PI_results.pkl               # 전체 결과
├── stage1_HC-PB_PB-PI_results_factor_scores.csv # 요인점수
├── stage1_HC-PB_PB-PI_results_loadings.csv      # 요인적재량
├── stage1_HC-PB_PB-PI_results_paths.csv         # 경로계수
├── stage1_HC-PB_PB-PI_results_fit_indices.csv   # 적합도
└── stage1_HC-PB_PB-PI_results_measurement_params.csv  # 측정모델 파라미터
```

**예상 시간:** 약 2-3분

---

### 3️⃣ **2단계 순차추정 (선택모델)** - 선택사항

**파일:** `examples/sequential_stage2_with_extended_model.py`

**현재 설정 확인:**
```python
STAGE1_RESULT_FILE = "stage1_HC-PB_PB-PI_results.pkl"
STANDARDIZATION_METHOD = 'zscore'
MAIN_LVS = ['purchase_intention', 'nutrition_knowledge']
LV_ATTRIBUTE_INTERACTIONS = [
    ('purchase_intention', 'health_label'),
    ('nutrition_knowledge', 'price')
]
```

**실행 방법:**
```bash
python examples/sequential_stage2_with_extended_model.py
```

**생성 파일:**
```
results/sequential_stage_wise/
└── st2_HC-PB_PB-PI1_PI_NK_int_PIxhl_NKxpr2_results.csv
```

**예상 시간:** 약 3-5분

---

## 📝 재실행 순서

### **권장 순서:**

```
1. CFA Only (측정모델만)
   ↓
2. 1단계 순차추정 (SEM)
   ↓
3. 2단계 순차추정 (선택모델) - 선택사항
```

### **이유:**

1. **CFA Only**
   - 측정모델 파라미터 확인
   - 요인적재량, 적합도 확인
   - 동시추정에서 초기값으로 사용

2. **1단계 순차추정**
   - 구조모델 경로계수 확인
   - 요인점수 생성 (원본 + 변환)
   - 2단계에서 사용

3. **2단계 순차추정**
   - 1단계 요인점수 사용
   - 선택모델 추정
   - 모델 비교

---

## 🔧 실행 스크립트

### **일괄 실행 스크립트 (선택사항)**

`run_update_to_328.py` 파일을 만들어서 한 번에 실행할 수 있습니다:

```python
"""
328명 데이터로 CFA 및 1단계 순차추정 재실행
"""
import subprocess
import sys
from pathlib import Path

print("=" * 70)
print("328명 데이터로 업데이트")
print("=" * 70)

# 1. CFA Only
print("\n[1/2] CFA Only 실행 중...")
result = subprocess.run(
    [sys.executable, "examples/sequential_cfa_only_example.py"],
    cwd=Path.cwd()
)
if result.returncode != 0:
    print("❌ CFA 실행 실패")
    sys.exit(1)
print("✅ CFA 완료")

# 2. 1단계 순차추정
print("\n[2/2] 1단계 순차추정 실행 중...")
result = subprocess.run(
    [sys.executable, "examples/sequential_stage1.py"],
    cwd=Path.cwd()
)
if result.returncode != 0:
    print("❌ 1단계 순차추정 실행 실패")
    sys.exit(1)
print("✅ 1단계 순차추정 완료")

print("\n" + "=" * 70)
print("✅ 모든 업데이트 완료!")
print("=" * 70)
```

**실행:**
```bash
python run_update_to_328.py
```

---

## ✅ 검증 방법

### **재실행 후 확인:**

```bash
python check_dataset_size.py
```

**예상 출력:**
```
현재 통합 데이터셋: 328명
✅ 328명 데이터 사용 중
✅ 1단계 순차추정: 328명 (일치)
✅ CFA: 328명 (일치)
```

---

## 📊 재실행 전후 비교

### **재실행 전 (현재)**

```
통합 데이터셋: 328명
CFA 결과: 326명 ❌
1단계 순차추정: 326명 ❌
동시추정: 328명 ✅
```

### **재실행 후 (예상)**

```
통합 데이터셋: 328명 ✅
CFA 결과: 328명 ✅
1단계 순차추정: 328명 ✅
동시추정: 328명 ✅
```

---

## ⚠️ 주의사항

### **1. 기존 결과 백업**

재실행 전에 기존 결과를 백업하는 것을 권장합니다:

```bash
# 백업 폴더 생성
mkdir -p results/sequential_stage_wise/backup_326

# 기존 결과 백업
cp results/sequential_stage_wise/cfa_results.pkl results/sequential_stage_wise/backup_326/
cp results/sequential_stage_wise/stage1_HC-PB_PB-PI_results.pkl results/sequential_stage_wise/backup_326/
```

### **2. 2단계 순차추정 재실행 여부**

- **재실행 필요:** 1단계 결과를 사용하는 모든 2단계 모델
- **재실행 불필요:** 동시추정 (이미 328명 데이터 사용)

### **3. 결과 파일 덮어쓰기**

- 동일한 파일명으로 저장되므로 기존 결과가 덮어쓰여집니다
- 백업이 필요한 경우 미리 백업하세요

---

## 🎯 최종 체크리스트

- [ ] 통합 데이터셋 확인 (328명)
- [ ] 기존 결과 백업 (선택사항)
- [ ] CFA Only 재실행
- [ ] 1단계 순차추정 재실행
- [ ] 2단계 순차추정 재실행 (선택사항)
- [ ] 결과 검증 (`check_dataset_size.py`)
- [ ] 요인점수 통계 확인 (평균 0, 표준편차 1)

---

## 📞 문제 발생 시

### **오류 메시지 확인:**

1. **데이터 로드 오류**
   - `integrated_data.csv` 파일 존재 확인
   - 파일 경로 확인

2. **개인 수 불일치**
   - `respondent_id` 컬럼 확인
   - 중복 ID 확인

3. **추정 실패**
   - 로그 파일 확인
   - 데이터 결측치 확인

---

## 📚 관련 문서

- `ESTIMATION_MODULES_REVIEW.md` - 3가지 추정 모듈 검토
- `STAGE1_FACTOR_SCORES_ANALYSIS.md` - 1단계 요인점수 분석
- `examples/STAGE1_USAGE_GUIDE.md` - 1단계 사용 가이드
- `examples/STAGE2_USAGE_EXAMPLES.md` - 2단계 사용 예시

---

**결론:** CFA와 1단계 순차추정을 재실행하여 328명 데이터로 업데이트하세요! 🎯

