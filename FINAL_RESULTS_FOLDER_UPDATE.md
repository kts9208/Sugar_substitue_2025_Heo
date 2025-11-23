# 최종 결과 폴더 통합 완료

**작성 날짜:** 2025-11-23  
**작성자:** ICLV Team

---

## 📋 작업 요약

### ✅ **완료된 작업**

모든 추정 방법의 결과 파일이 **`results/final/`** 폴더에 저장되도록 코드를 수정했습니다.

---

## 📁 최종 결과 폴더 구조

```
results/final/
├── README.md                           # 폴더 설명
│
├── cfa_only/                          # CFA Only (측정모델만)
│   └── (cfa_results.pkl 등)
│
├── choice_only/                       # Choice Only (선택모델만)
│   └── (choice_model_results.csv 등)
│
├── sequential/                        # 순차추정
│   ├── stage1/                       # 1단계 (SEM)
│   │   └── (stage1_*.pkl 등)
│   │
│   └── stage2/                       # 2단계 (선택모델)
│       └── (st2_*.csv 등)
│
└── simultaneous/                      # 동시추정
    ├── results/                      # 추정 결과
    │   ├── simultaneous_*_results_*.csv
    │   └── simultaneous_*_results_*.npy
    │
    └── logs/                         # 로그 파일
        ├── simultaneous_estimation_log_*.txt
        └── simultaneous_estimation_log_*_params_grads.csv
```

---

## 🔧 수정된 파일

### **1. CFA Only**

**파일:** `examples/sequential_cfa_only_example.py`

**변경 사항:**
```python
# 이전
save_path = project_root / "results" / "sequential_stage_wise" / "cfa_results.pkl"

# 이후
save_dir = project_root / "results" / "final" / "cfa_only"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "cfa_results.pkl"
```

**저장 위치:** `results/final/cfa_only/`

---

### **2. Choice Only**

**파일:** `scripts/test_choice_model.py`

**변경 사항:**
- 결과 저장 기능 추가 (이전에는 화면 출력만)
- CSV 파일로 저장

**저장 위치:** `results/final/choice_only/`

**저장 파일:**
- `choice_model_results.csv` - 추정 결과

---

### **3. 순차추정 1단계**

**파일:** `examples/sequential_stage1.py`

**변경 사항:**
```python
# 이전
save_path = project_root / "results" / "sequential_stage_wise" / f"stage1_{path_name}_results.pkl"

# 이후
save_dir = project_root / "results" / "final" / "sequential" / "stage1"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / f"stage1_{path_name}_results.pkl"
```

**저장 위치:** `results/final/sequential/stage1/`

---

### **4. 순차추정 2단계**

**파일:** `examples/sequential_stage2_with_extended_model.py`

**변경 사항:**
```python
# 이전 (결과 저장)
save_dir = project_root / "results" / "sequential_stage_wise"

# 이후 (결과 저장)
save_dir = project_root / "results" / "final" / "sequential" / "stage2"

# 이전 (1단계 결과 로드)
stage1_path = project_root / "results" / "sequential_stage_wise" / STAGE1_RESULT_FILE

# 이후 (1단계 결과 로드)
stage1_path = project_root / "results" / "final" / "sequential" / "stage1" / STAGE1_RESULT_FILE
```

**저장 위치:** `results/final/sequential/stage2/`

**로드 위치:** `results/final/sequential/stage1/` (1단계 결과)

---

### **5. 동시추정**

**파일:** `scripts/test_gpu_batch_iclv.py`

**변경 사항:**
```python
# 이전 (결과 파일)
output_dir = project_root / 'results'

# 이후 (결과 파일)
output_dir = project_root / 'results' / 'final' / 'simultaneous' / 'results'

# 이전 (로그 파일)
log_file = project_root / 'results' / f'simultaneous_estimation_log_{timestamp}.txt'

# 이후 (로그 파일)
log_dir = project_root / 'results' / 'final' / 'simultaneous' / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f'simultaneous_estimation_log_{timestamp}.txt'
```

**저장 위치:**
- 결과 파일: `results/final/simultaneous/results/`
- 로그 파일: `results/final/simultaneous/logs/`

---

## 📊 추정 방법별 실행 파일 및 결과

| 추정 방법 | 실행 파일 | 결과 저장 위치 |
|----------|----------|---------------|
| **CFA Only** | `examples/sequential_cfa_only_example.py` | `results/final/cfa_only/` |
| **Choice Only** | `scripts/test_choice_model.py` | `results/final/choice_only/` |
| **순차추정 1단계** | `examples/sequential_stage1.py` | `results/final/sequential/stage1/` |
| **순차추정 2단계** | `examples/sequential_stage2_with_extended_model.py` | `results/final/sequential/stage2/` |
| **동시추정** | `scripts/test_gpu_batch_iclv.py` | `results/final/simultaneous/results/` (결과)<br>`results/final/simultaneous/logs/` (로그) |

---

## 🎯 사용 방법

### **1. 기존 결과 파일 이동 (선택사항)**

기존 `results/sequential_stage_wise/` 폴더의 결과를 새 폴더로 이동하려면:

```bash
# CFA 결과 이동
cp results/sequential_stage_wise/cfa_results* results/final/cfa_only/

# 1단계 결과 이동
cp results/sequential_stage_wise/stage1_* results/final/sequential/stage1/

# 2단계 결과 이동
cp results/sequential_stage_wise/st2_* results/final/sequential/stage2/
```

### **2. 새로 추정 실행**

이제 각 추정 코드를 실행하면 자동으로 `results/final/` 폴더에 저장됩니다:

```bash
# CFA Only
python examples/sequential_cfa_only_example.py

# 1단계 순차추정
python examples/sequential_stage1.py

# 2단계 순차추정
python examples/sequential_stage2_with_extended_model.py

# 동시추정
python scripts/test_gpu_batch_iclv.py

# Choice Only (테스트)
python scripts/test_choice_model.py
```

---

## ⚠️ 주의사항

### **1. 순차추정 2단계 실행 전**

2단계를 실행하기 전에 **반드시 1단계를 먼저 실행**해야 합니다.

2단계는 `results/final/sequential/stage1/` 폴더에서 1단계 결과를 로드합니다.

### **2. 기존 결과 백업**

기존 `results/sequential_stage_wise/` 폴더의 결과를 보존하려면 백업하세요:

```bash
# 백업 폴더 생성
mkdir -p results/backup_old_structure

# 기존 결과 백업
cp -r results/sequential_stage_wise/* results/backup_old_structure/
```

### **3. 동시추정 파일 누적**

동시추정은 타임스탬프별로 파일이 누적되므로 주기적으로 정리가 필요합니다.

---

## 📚 관련 문서

- `results/final/README.md` - 최종 결과 폴더 설명
- `DATASET_UPDATE_SUMMARY.md` - 328명 데이터 업데이트 요약
- `UPDATE_TO_328_INDIVIDUALS.md` - 328명 업데이트 가이드
- `ESTIMATION_MODULES_REVIEW.md` - 3가지 추정 모듈 검토

---

## ✅ 체크리스트

- [x] 최종 결과 폴더 구조 생성
- [x] CFA Only 코드 수정
- [x] Choice Only 코드 수정 (결과 저장 기능 추가)
- [x] 순차추정 1단계 코드 수정
- [x] 순차추정 2단계 코드 수정 (저장 + 로드 경로)
- [x] 동시추정 코드 수정 (결과 + 로그 경로)
- [x] README 파일 작성
- [ ] 기존 결과 파일 이동 (선택사항)
- [ ] 새로 추정 실행 (328명 데이터)

---

**모든 추정 결과가 이제 `results/final/` 폴더에 체계적으로 저장됩니다!** 🎯

