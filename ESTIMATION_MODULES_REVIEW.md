# 3가지 추정 모듈 검토 결과

**검토 날짜:** 2025-11-22  
**검토자:** ICLV Team

---

## 📋 요약

현재 3가지 추정 모듈이 구축되어 있으며, 각각의 실행 코드와 결과 저장소를 검토한 결과입니다.

| 추정 방법 | 실행 파일 | 결과 저장소 | 상태 |
|---------|---------|-----------|------|
| **선택모델만** | `test_choice_model.py` (테스트용) | `results/choice_model_only/` | ⚠️ 개선 필요 |
| **순차추정** | `sequential_stage1.py`, `sequential_stage2_with_extended_model.py` | `results/sequential_stage_wise/` | ✅ 완벽 |
| **동시추정** | `test_gpu_batch_iclv.py` | `results/` (루트에 분산) | ✅ 작동, 정리 필요 |

---

## 1️⃣ 선택모델만 (Choice Model Only)

### 실행 파일
- **테스트용:** `scripts/test_choice_model.py`
  - 샘플 데이터로 선택모델 fit() 메서드 테스트
  - 100명 개인, 3개 대안 (제품A, 제품B, 구매안함)
  - 요인점수 3개 (purchase_intention, perceived_price, nutrition_knowledge)

- **실제 데이터용:** ❌ 없음
  - 순차추정 2단계가 사실상 이 역할을 함

### 결과 저장소
```
results/choice_model_only/
├── model_summary.txt
└── parameter_statistics.csv
```

### 평가
- ⚠️ **개선 필요:** 실제 데이터용 독립 실행 파일 부재
- ✅ **대안:** 순차추정 2단계를 선택모델만 추정으로 사용 가능

---

## 2️⃣ 순차추정 (Sequential Estimation)

### 실행 파일

#### 1단계 (SEM: 측정모델 + 구조모델)
**파일:** `examples/sequential_stage1.py`

**주요 기능:**
- ✅ 경로 설정: True/False로 간단하게 켜고 끄기
- ✅ 자동 파일명 생성: `stage1_{경로명}_results.*`
- ✅ 수정지수 계산 옵션
- ✅ 요인점수 변환 방법 선택 (zscore/center)

**설정 예시:**
```python
PATHS = {
    'HC->PB': True,   # 건강관심도 → 건강유익성
    'HC->PP': False,  # 건강관심도 → 가격수준
    'HC->PI': False,  # 건강관심도 → 구매의도
    'PB->PI': True,   # 건강유익성 → 구매의도
    'PP->PI': False,  # 가격수준 → 구매의도
    'NK->PI': False,  # 영양지식 → 구매의도
}
```

#### 2단계 (선택모델)
**파일:** `examples/sequential_stage2_with_extended_model.py`

**주요 기능:**
- ✅ 1단계 결과 자동 로드
- ✅ 주효과, 조절효과, 상호작용 모두 지원
- ✅ 자동 파일명 생성: `st2_{1단계경로}1_{2단계설정}2_results.csv`
- ✅ 모델 비교 지원

**설정 예시:**
```python
STAGE1_RESULT_FILE = "stage1_HC-PB_PB-PI_results.pkl"
MAIN_LVS = ['purchase_intention', 'nutrition_knowledge']
LV_ATTRIBUTE_INTERACTIONS = [
    ('purchase_intention', 'health_label'),
    ('nutrition_knowledge', 'price')
]
```

#### 통합 실행
**파일:** `scripts/test_sequential_estimation.py`
- 1단계 + 2단계 통합 실행
- 모든 LV 주효과 추정

### 결과 저장소
```
results/sequential_stage_wise/
├── cfa_results.pkl                              # CFA 결과 (측정모델만)
├── cfa_results_loadings.csv
├── cfa_results_fit_indices.csv
│
├── stage1_{경로명}_results.pkl                  # 1단계 전체 결과
├── stage1_{경로명}_results_factor_scores.csv   # 요인점수
├── stage1_{경로명}_results_loadings.csv        # 요인적재량
├── stage1_{경로명}_results_paths.csv           # 경로계수
├── stage1_{경로명}_results_fit_indices.csv     # 적합도
│
├── st2_{1단계}1_{2단계}2_results.csv            # 2단계 통합 결과
│
├── model_comparison_summary.csv                 # 모델 비교
└── likelihood_ratio_test_results.csv            # 우도비 검정
```

### 파일명 규칙
- **1단계:** `stage1_{경로명}_results.*`
  - 예: `stage1_HC-PB_PB-PI_results.pkl`
  
- **2단계:** `st2_{1단계경로}1_{2단계설정}2_results.csv`
  - 예: `st2_HC-PB_PB-PI1_PI_NK2_results.csv`
  - `1`: 1단계 경로 정보
  - `2`: 2단계 선택모델 설정

### 평가
- ✅ **완벽:** 1단계, 2단계 모두 독립 실행 가능
- ✅ **체계적:** 파일명 규칙 명확, 결과 저장 체계적
- ✅ **유연성:** 다양한 모델 설정 지원

---

## 3️⃣ 동시추정 (Simultaneous Estimation)

### 실행 파일
**파일:** `scripts/test_gpu_batch_iclv.py`

**주요 기능:**
- ✅ CSV 파일명에서 자동 설정 파싱
- ✅ CFA 결과 자동 로드 (측정모델 고정)
- ✅ GPU 배치 처리
- ✅ 구조모델 + 선택모델 동시 추정

**설정 예시:**
```python
# 순차추정 2단계 CSV 파일명만 지정
INITIAL_PARAMS_CSV = 'st2_HC-PB_PB-PI1_PI2_results.csv'

# 자동으로 파싱됨:
# - 1단계 경로: HC->PB, PB->PI
# - 선택모델: PI 주효과
```

### 결과 저장소
```
results/
├── simultaneous_{경로명}_{선택모델LV}_results_{timestamp}.csv
├── simultaneous_{경로명}_{선택모델LV}_results_{timestamp}.npy
├── simultaneous_estimation_log_{timestamp}.txt
└── simultaneous_estimation_log_{timestamp}_params_grads.csv
```

### 파일명 규칙
- **결과:** `simultaneous_{경로명}_{선택모델LV}_results_{timestamp}.csv`
  - 예: `simultaneous_HC-PB_PB-PI_results_20251122_165016.csv`
  
- **로그:** `simultaneous_estimation_log_{timestamp}.txt`

### 평가
- ✅ **완벽:** 자동 설정 파싱, GPU 가속
- ⚠️ **정리 필요:** 결과 파일이 루트에 분산됨

---

## 🔍 발견된 문제점

### 1. 선택모델만 실행 파일 부재
- `test_choice_model.py`는 샘플 데이터용
- 실제 데이터로 선택모델만 추정하는 독립 실행 파일이 없음
- **해결책:** 순차추정 2단계가 사실상 이 역할을 하므로 문제 없음

### 2. 동시추정 결과 파일 위치
- 동시추정 결과가 `results/` 루트에 분산되어 있음
- 순차추정처럼 별도 폴더가 없음
- **제안:** `results/simultaneous/` 폴더 생성 권장

### 3. 파일명 규칙 불일치
- 순차추정: `stage1_`, `st2_` 접두사
- 동시추정: `simultaneous_` 접두사 + 타임스탬프
- **제안:** 일관성 있는 명명 규칙 필요

---

## ✅ 권장 개선 사항

### 1. 결과 저장소 재구성

**제안 구조:**
```
results/
├── choice_model_only/              # 선택모델만
│   └── (현재 유지)
│
├── sequential/                     # 순차추정 (이름 변경)
│   ├── stage1/                    # 1단계 결과
│   │   ├── cfa_results.pkl
│   │   └── stage1_*.pkl
│   ├── stage2/                    # 2단계 결과
│   │   └── st2_*.csv
│   └── comparison/                # 모델 비교
│       ├── model_comparison_summary.csv
│       └── likelihood_ratio_test_results.csv
│
└── simultaneous/                   # 동시추정 (신규 폴더)
    ├── results/
    │   ├── simultaneous_*.csv
    │   └── simultaneous_*.npy
    └── logs/
        ├── simultaneous_estimation_log_*.txt
        └── simultaneous_estimation_log_*_params_grads.csv
```

### 2. 파일명 규칙 통일

**순차추정:**
- 1단계: `seq_stage1_{경로명}_{timestamp}.pkl`
- 2단계: `seq_stage2_{1단계}_{2단계}_{timestamp}.csv`

**동시추정:**
- 결과: `sim_{경로명}_{선택모델}_{timestamp}.csv`
- 로그: `sim_log_{timestamp}.txt`

### 3. 선택모델만 실행 파일 추가 (선택사항)

**파일:** `scripts/run_choice_model_only.py`
- 요인점수 CSV 파일 로드
- 선택모델만 추정
- 결과 저장: `results/choice_model_only/`

---

## 📊 최종 평가

| 항목 | 선택모델만 | 순차추정 | 동시추정 |
|-----|----------|---------|---------|
| **실행 파일** | ⚠️ 테스트용만 | ✅ 완벽 | ✅ 완벽 |
| **결과 저장** | ✅ 정리됨 | ✅ 체계적 | ⚠️ 분산됨 |
| **파일명 규칙** | ✅ 명확 | ✅ 명확 | ✅ 명확 |
| **자동화** | - | ✅ 높음 | ✅ 매우 높음 |
| **문서화** | ⚠️ 부족 | ✅ 충분 | ✅ 충분 |

**종합 평가:** ✅ **전반적으로 우수**, 일부 정리 필요

---

## 🎯 다음 단계

1. ✅ **즉시 가능:** 현재 상태로 모든 추정 실행 가능
2. 📁 **권장:** 동시추정 결과 폴더 정리 (`results/simultaneous/`)
3. 📝 **선택:** 파일명 규칙 통일 (기존 파일 영향 없음)
4. 🔧 **선택:** 선택모델만 실행 파일 추가

---

**결론:** 3가지 추정 모듈 모두 정상 작동하며, 결과 저장소도 대부분 체계적입니다. 동시추정 결과 파일 정리만 하면 완벽합니다.

