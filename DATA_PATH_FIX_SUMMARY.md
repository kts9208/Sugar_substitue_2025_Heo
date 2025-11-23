# 데이터 경로 수정 완료

**작성 날짜:** 2025-11-23  
**작성자:** ICLV Team

---

## 📋 문제 상황

### **오류 메시지**
```
FileNotFoundError: [Errno 2] No such file or directory: 
'...\integrated_data_cleaned.csv'
```

### **원인**
일부 예제 파일이 삭제된 326명 백업 파일(`integrated_data_cleaned.csv`)을 참조하고 있었습니다.

---

## ✅ 수정 완료

### **수정된 파일 (4개)**

| 파일 | 이전 경로 | 수정 후 경로 |
|------|----------|-------------|
| `examples/sequential_cfa_only_example.py` | `integrated_data_cleaned.csv` | `integrated_data.csv` ✅ |
| `examples/bootstrap_sequential_example.py` (3곳) | `integrated_data_cleaned.csv` | `integrated_data.csv` ✅ |
| `examples/choice_model_only.py` | `integrated_data_cleaned.csv` | `integrated_data.csv` ✅ |

### **추가 수정 (bootstrap 예제)**

`examples/bootstrap_sequential_example.py`의 1단계 결과 로드 경로도 수정:

```python
# 이전
results_dir = project_root / "results" / "sequential_stage_wise"

# 이후
results_dir = project_root / "results" / "final" / "sequential" / "stage1"
```

---

## ✅ 검증 완료

### **CFA Only 실행 성공**

```bash
python examples/sequential_cfa_only_example.py
```

**결과:**
- ✅ 데이터 로드 완료: 5,904행, 60열
- ✅ **개인 수: 328명** (정상)
- ✅ CFA 추정 완료
- ✅ 결과 저장: `results/final/cfa_only/`

**적합도 지수:**
- CFI: 0.8388
- TLI: 0.8270
- RMSEA: 0.0679
- AIC: 161.98
- BIC: 488.18

**유의한 상관관계:**
- `perceived_benefit ↔ purchase_intention`: 0.7344 (p<0.001) ***
- `perceived_benefit ↔ health_concern`: 0.3386 (p<0.001) ***

---

## 📊 현재 상태

### **데이터 파일**

| 파일 | 개인 수 | 상태 |
|------|---------|------|
| `integrated_data.csv` | **328명** | ✅ 사용 중 |
| `integrated_data_cleaned.csv` | - | ❌ 삭제됨 |
| `integrated_data_backup.csv` | - | ❌ 삭제됨 |

### **예제 파일 데이터 경로**

| 파일 | 데이터 경로 | 상태 |
|------|------------|------|
| `sequential_cfa_only_example.py` | `integrated_data.csv` | ✅ 정상 |
| `sequential_stage1.py` | `integrated_data.csv` | ✅ 정상 |
| `sequential_stage2_with_extended_model.py` | `integrated_data.csv` | ✅ 정상 |
| `bootstrap_sequential_example.py` | `integrated_data.csv` | ✅ 정상 |
| `choice_model_only.py` | `integrated_data.csv` | ✅ 정상 |
| `correlation_analysis_example.py` | `integrated_data.csv` | ✅ 정상 |

---

## 🎯 다음 단계

### **1. 1단계 순차추정 실행 (328명)**

```bash
python examples/sequential_stage1.py
```

**예상 결과:**
- 개인 수: 328명
- 저장 위치: `results/final/sequential/stage1/`

### **2. 2단계 순차추정 실행 (선택사항)**

```bash
python examples/sequential_stage2_with_extended_model.py
```

**주의:** 1단계 실행 후 진행

### **3. 검증**

```bash
python check_dataset_size.py
```

**예상 출력:**
```
현재 통합 데이터셋: 328명
✅ CFA: 328명 (일치)
✅ 1단계 순차추정: 328명 (일치)
```

---

## 📚 관련 문서

- `SUMMARY_FINAL_RESULTS_MIGRATION.md` - 최종 결과 폴더 통합 요약
- `FINAL_RESULTS_FOLDER_UPDATE.md` - 최종 결과 폴더 업데이트 가이드
- `DATASET_UPDATE_SUMMARY.md` - 328명 데이터 업데이트 요약
- `check_data_paths.py` - 데이터 경로 확인 스크립트

---

## ✅ 체크리스트

- [x] 데이터 경로 오류 확인
- [x] `sequential_cfa_only_example.py` 수정
- [x] `bootstrap_sequential_example.py` 수정 (3곳)
- [x] `choice_model_only.py` 수정
- [x] CFA Only 실행 검증 (328명)
- [ ] 1단계 순차추정 실행 (328명)
- [ ] 2단계 순차추정 실행 (선택사항)
- [ ] 최종 검증

---

**모든 예제 파일이 328명 데이터(`integrated_data.csv`)를 사용하도록 수정 완료!** 🎯

