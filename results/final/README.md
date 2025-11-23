# 최종 결과 폴더 (Final Results)

**생성 날짜:** 2025-11-23  
**목적:** 모든 추정 방법의 최종 결과를 한 곳에 저장

---

## 📁 폴더 구조

```
results/final/
├── README.md                           # 이 파일
│
├── cfa_only/                          # CFA Only (측정모델만)
│   ├── cfa_results.pkl
│   ├── cfa_results_factor_scores.csv
│   ├── cfa_results_loadings.csv
│   ├── cfa_results_fit_indices.csv
│   └── ...
│
├── choice_only/                       # Choice Only (선택모델만)
│   ├── choice_model_results.csv
│   ├── choice_model_summary.txt
│   └── ...
│
├── sequential/                        # 순차추정
│   ├── stage1/                       # 1단계 (SEM)
│   │   ├── stage1_HC-PB_PB-PI_results.pkl
│   │   ├── stage1_HC-PB_PB-PI_results_factor_scores.csv
│   │   └── ...
│   │
│   └── stage2/                       # 2단계 (선택모델)
│       ├── st2_HC-PB_PB-PI1_PI_NK2_results.csv
│       └── ...
│
└── simultaneous/                      # 동시추정
    ├── results/
    │   ├── simultaneous_HC-PB_PB-PI_results_YYYYMMDD_HHMMSS.csv
    │   └── simultaneous_HC-PB_PB-PI_results_YYYYMMDD_HHMMSS.npy
    │
    └── logs/
        ├── simultaneous_estimation_log_YYYYMMDD_HHMMSS.txt
        └── simultaneous_estimation_log_YYYYMMDD_HHMMSS_params_grads.csv
```

---

## 📊 추정 방법별 결과 파일

### 1️⃣ CFA Only (측정모델만)

**실행 파일:** `examples/sequential_cfa_only_example.py`

**결과 파일:**
- `cfa_results.pkl` - 전체 결과
- `cfa_results_factor_scores.csv` - 요인점수
- `cfa_results_loadings.csv` - 요인적재량
- `cfa_results_fit_indices.csv` - 적합도 지수
- `cfa_results_all_params.csv` - 모든 파라미터
- `cfa_results_measurement_params.csv` - 측정모델 파라미터
- `cfa_results_correlation_matrix.csv` - 상관행렬
- `cfa_results_pvalue_matrix.csv` - p-value 행렬

---

### 2️⃣ Choice Only (선택모델만)

**실행 파일:** `scripts/test_choice_model.py` (테스트용)

**결과 파일:**
- `choice_model_results.csv` - 추정 결과
- `choice_model_summary.txt` - 요약 통계

---

### 3️⃣ 순차추정 (Sequential Estimation)

#### **1단계 (SEM: 측정모델 + 구조모델)**

**실행 파일:** `examples/sequential_stage1.py`

**결과 파일:**
- `stage1_{경로명}_results.pkl` - 전체 결과
- `stage1_{경로명}_results_factor_scores.csv` - 요인점수
- `stage1_{경로명}_results_loadings.csv` - 요인적재량
- `stage1_{경로명}_results_paths.csv` - 경로계수
- `stage1_{경로명}_results_fit_indices.csv` - 적합도 지수
- `stage1_{경로명}_results_measurement_params.csv` - 측정모델 파라미터

#### **2단계 (선택모델)**

**실행 파일:** `examples/sequential_stage2_with_extended_model.py`

**결과 파일:**
- `st2_{1단계경로}1_{2단계설정}2_results.csv` - 통합 결과

---

### 4️⃣ 동시추정 (Simultaneous Estimation)

**실행 파일:** `scripts/test_gpu_batch_iclv.py`

**결과 파일:**
- `simultaneous_{경로명}_{선택모델LV}_results_{timestamp}.csv` - 파라미터
- `simultaneous_{경로명}_{선택모델LV}_results_{timestamp}.npy` - raw params
- `simultaneous_estimation_log_{timestamp}.txt` - 로그
- `simultaneous_estimation_log_{timestamp}_params_grads.csv` - 파라미터/그래디언트

---

## 🎯 사용 방법

### **결과 파일 찾기**

1. **CFA만 필요한 경우:** `results/final/cfa_only/`
2. **순차추정 1단계 결과:** `results/final/sequential/stage1/`
3. **순차추정 2단계 결과:** `results/final/sequential/stage2/`
4. **동시추정 결과:** `results/final/simultaneous/results/`

### **최신 결과 확인**

- 순차추정: 파일명에 경로 정보 포함
- 동시추정: 타임스탬프로 정렬하여 최신 파일 확인

---

## 📝 주의사항

1. **기존 결과 백업**
   - 재실행 시 동일한 파일명으로 덮어쓰여집니다
   - 중요한 결과는 별도 백업 권장

2. **파일명 규칙**
   - 순차추정: 경로 정보 포함 (예: `stage1_HC-PB_PB-PI_results.pkl`)
   - 동시추정: 타임스탬프 포함 (예: `simultaneous_HC-PB_PB-PI_results_20251123_120000.csv`)

3. **디스크 공간**
   - 동시추정은 타임스탬프별로 누적되므로 주기적으로 정리 필요

---

**모든 최종 결과는 이 폴더에 저장됩니다!** 🎯

