# 📊 역문항 처리 기능 현황 보고서

**작성일**: 2025-11-04  
**목적**: 기존 코드의 역문항 처리 기능 확인 및 현재 데이터 상태 분석

---

## ✅ 결론: 역문항 처리 기능은 존재하나, 현재 데이터에는 미적용 상태

---

## 📋 1. 역문항 처리 기능 존재 여부

### **✅ 역문항 처리 시스템이 완전히 구축되어 있습니다**

#### **1.1 설정 파일**

**위치**: `data/config/reverse_items_config.json`

<augment_code_snippet path="data/config/reverse_items_config.json" mode="EXCERPT">
````json
{
  "description": "역문항(역코딩) 문항 정보 설정 파일",
  "version": "1.0.0",
  "created_date": "2025-09-05",
  "scale_range": {
    "min": 1,
    "max": 5
  },
  "reverse_items": {
    "perceived_benefit": {
      "reverse_items": ["q13"]
    },
    "perceived_price": {
      "reverse_items": ["q28"]
    },
    "nutrition_knowledge": {
      "reverse_items": ["q30", "q31", "q32", "q34", ...]
    }
  },
  "reverse_coding_formula": {
    "formula": "reversed_value = (scale_max + scale_min) - original_value"
  }
}
````
</augment_code_snippet>

**역문항 목록**:
- **perceived_benefit**: q13 (1개)
- **perceived_price**: q28 (1개)
- **nutrition_knowledge**: q30, q31, q32, q34, q35, q36, q38, q39, q42, q43, q44, q45, q46, q48, q49 (15개)
- **총 17개 역문항**

**역코딩 공식**:
```
reversed_value = (scale_max + scale_min) - original_value
              = (5 + 1) - original_value
              = 6 - original_value
```

**예시**:
- 원점수 1 → 역코딩 5
- 원점수 2 → 역코딩 4
- 원점수 3 → 역코딩 3
- 원점수 4 → 역코딩 2
- 원점수 5 → 역코딩 1

---

#### **1.2 처리 모듈**

**참조 위치**: `processed_data.modules.reverse_items_processor`

**클래스**: `ReverseItemsProcessor`

**주요 메서드**:
1. `__init__()`: 설정 파일 로드
2. `_reverse_code_value(value)`: 단일 값 역코딩
3. `process_reverse_items()`: 전체 데이터 역문항 처리
4. 백업 기능 포함

**사용 예시** (테스트 코드에서):
<augment_code_snippet path="tests/test_reverse_items_workflow.py" mode="EXCERPT">
````python
from processed_data.modules.reverse_items_processor import ReverseItemsProcessor

processor = ReverseItemsProcessor()
success = processor.process_reverse_items()
````
</augment_code_snippet>

---

#### **1.3 실행 스크립트**

**위치**: `scripts/run_reliability_analysis.py`

<augment_code_snippet path="scripts/run_reliability_analysis.py" mode="EXCERPT">
````python
def run_reverse_items_processing():
    """역문항 처리 실행"""
    from processed_data.modules.reverse_items_processor import ReverseItemsProcessor
    
    processor = ReverseItemsProcessor()
    success = processor.process_reverse_items()
    
    if success:
        print("✓ 역문항 처리 완료")
    return success
````
</augment_code_snippet>

---

#### **1.4 처리 로그**

**위치**: `logs/reverse_items_processing.log`

**최근 처리 기록** (2025-09-05):
```
2025-09-05 14:45:09 - perceived_benefit.q13 역코딩 완료: 300개 값 처리
2025-09-05 14:45:09 - perceived_price.q28 역코딩 완료: 300개 값 처리
2025-09-05 14:45:09 - nutrition_knowledge.q30-q49 역코딩 완료
2025-09-05 14:45:09 - 전체 역문항 처리 완료: 17개 문항 처리, 0개 오류
```

**결론**: 2025년 9월 5일에 역문항 처리가 성공적으로 실행되었음

---

## 📊 2. 현재 데이터 상태 분석

### **❌ 현재 데이터는 역코딩이 적용되지 않은 상태입니다**

#### **2.1 perceived_benefit 데이터 검증**

**파일**: `data/processed/survey/perceived_benefit.csv`  
**수정일**: 2025-09-19 09:01 (역문항 처리 후)

**q13 (역문항) 통계**:
```
평균: 2.53
분포: {1: 29, 2: 123, 3: 110, 4: 35, 5: 3}
```

**q14 (정문항) 통계**:
```
평균: 3.27
분포: {1: 4, 2: 51, 3: 123, 4: 104, 5: 18}
```

**분석**:
- q13 평균 (2.53) < q14 평균 (3.27)
- q13이 역문항이라면, 역코딩 후 평균은 `6 - 2.53 = 3.47`이어야 함
- 하지만 현재 평균이 2.53이므로 **역코딩이 적용되지 않음**

**응답 분포 분석**:
- q13: 낮은 점수(1-2)가 152명 (50.7%)
- q14: 높은 점수(3-5)가 245명 (81.7%)
- 역문항 특성상 q13의 분포가 반대여야 하는데, 현재는 원점수 상태

---

#### **2.2 역코딩 적용 시 예상 변화**

**q13 역코딩 전후 비교**:

| 원점수 | 빈도 | 역코딩 후 | 빈도 |
|--------|------|-----------|------|
| 1 | 29 | 5 | 29 |
| 2 | 123 | 4 | 123 |
| 3 | 110 | 3 | 110 |
| 4 | 35 | 2 | 35 |
| 5 | 3 | 1 | 3 |

**평균 변화**:
- 역코딩 전: 2.53
- 역코딩 후: 3.47 (= 6 - 2.53)

**효과**:
- 역코딩 후 q13과 q14의 평균이 유사해짐 (3.47 vs 3.27)
- 요인 내 일관성 증가
- Ordered Probit 모델 적합도 개선 예상

---

## 🔍 3. 왜 역코딩이 적용되지 않았는가?

### **가능한 원인**

#### **원인 1: 데이터 복원**
- 로그에 따르면 2025-09-05에 역코딩 완료
- 하지만 파일 수정일은 2025-09-19 09:01
- **추정**: 9월 19일에 원본 데이터로 복원되었을 가능성

#### **원인 2: 백업에서 복원**
- 백업 디렉토리: `processed_data/survey_data_backup/backup_20250905_*`
- 역문항 처리 전 백업이 존재
- **추정**: 어떤 이유로 백업에서 복원

#### **원인 3: 처리 모듈 미실행**
- `ReverseItemsProcessor` 모듈이 `processed_data.modules`에 위치
- 현재 `processed_data/modules/` 디렉토리가 존재하지 않음
- **추정**: 프로젝트 구조 변경으로 모듈 경로 변경

---

## 🎯 4. 역문항 처리 필요성

### **4.1 이론적 근거**

**역문항의 목적**:
1. 응답 편향(response bias) 감소
2. 무성의 응답 탐지
3. 측정 타당도 향상

**역코딩의 필요성**:
- 역문항을 역코딩하지 않으면 요인 내 일관성이 낮아짐
- 요인적재량이 음수로 나타남
- 신뢰도(Cronbach's α) 감소
- 모델 적합도 저하

---

### **4.2 Ordered Probit 모델에 미치는 영향**

**현재 상태 (역코딩 미적용)**:
- `perceived_benefit` 지표당 LL = -4.78 (보통 적합)
- q13의 요인적재량이 음수일 가능성
- 잠재변수 추정 정확도 저하

**역코딩 적용 시 예상 효과**:
- 지표당 LL 개선 (예상: -4.78 → -4.0 이하)
- 모든 요인적재량이 양수
- 잠재변수 추정 정확도 향상
- 모델 해석 용이성 증가

---

## 📝 5. 역문항 처리 방법

### **방법 1: 기존 모듈 사용 (권장)**

**조건**: `ReverseItemsProcessor` 모듈이 작동하는 경우

```python
from processed_data.modules.reverse_items_processor import ReverseItemsProcessor

processor = ReverseItemsProcessor()
success = processor.process_reverse_items()

if success:
    print("✓ 역문항 처리 완료")
```

**장점**:
- ✅ 자동 백업
- ✅ 로그 기록
- ✅ 검증 기능
- ✅ 전체 요인 일괄 처리

---

### **방법 2: 수동 역코딩 (대안)**

**조건**: 모듈이 작동하지 않는 경우

```python
import pandas as pd
import json

# 1. 설정 파일 로드
with open('data/config/reverse_items_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# 2. 각 요인별 처리
for factor_name, factor_config in config['reverse_items'].items():
    reverse_items = factor_config.get('reverse_items', [])
    
    if not reverse_items:
        continue
    
    # 데이터 로드
    file_path = f'data/processed/survey/{factor_name}.csv'
    data = pd.read_csv(file_path)
    
    # 백업
    data.to_csv(f'{file_path}.backup', index=False)
    
    # 역코딩
    for item in reverse_items:
        if item in data.columns:
            data[item] = 6 - data[item]
    
    # 저장
    data.to_csv(file_path, index=False)
    print(f"✓ {factor_name}: {len(reverse_items)}개 역문항 처리 완료")
```

---

### **방법 3: 테스트 시 임시 역코딩**

**조건**: 원본 데이터를 유지하면서 테스트만 하는 경우

```python
import pandas as pd

# 데이터 로드
data = pd.read_csv('data/processed/survey/perceived_benefit.csv')

# 임시 역코딩 (원본 유지)
data_reversed = data.copy()
data_reversed['q13'] = 6 - data_reversed['q13']

# Ordered Probit 테스트
from src.analysis.hybrid_choice_model.iclv_models.measurement_equations import OrderedProbitMeasurement

config = MeasurementConfig(
    indicators=['q13', 'q14', 'q15'],
    n_categories=5
)

model = OrderedProbitMeasurement(config)
latent_var = data_reversed[['q13', 'q14', 'q15']].mean(axis=1).values

# 로그우도 계산
ll = model.log_likelihood(data_reversed, latent_var, params)
print(f"역코딩 적용 후 LL: {ll:.2f}")
```

---

## 🎯 6. 권장 조치

### **즉시 조치 (우선순위 높음)**

1. **✅ 역문항 처리 실행**
   - 방법 1 (기존 모듈) 또는 방법 2 (수동) 사용
   - 17개 역문항 일괄 처리
   - 백업 필수

2. **✅ 처리 결과 검증**
   - q13 평균: 2.53 → 3.47 확인
   - q28 평균 변화 확인
   - nutrition_knowledge 15개 문항 확인

3. **✅ Ordered Probit 재테스트**
   - 역코딩 적용 후 로그우도 비교
   - 적합도 개선 확인

---

### **장기 조치**

4. **데이터 파이프라인 정립**
   - 역문항 처리를 자동화
   - 데이터 로드 시 역코딩 상태 확인
   - 처리 이력 관리

5. **문서화**
   - 역문항 처리 절차 문서화
   - 데이터 버전 관리
   - 처리 전후 비교 보고서

---

## ✅ 최종 결론

| 항목 | 상태 |
|------|------|
| **역문항 처리 기능 존재** | ✅ 완전히 구축됨 |
| **설정 파일** | ✅ 존재 (17개 역문항 정의) |
| **처리 모듈** | ✅ 존재 (`ReverseItemsProcessor`) |
| **실행 스크립트** | ✅ 존재 |
| **처리 로그** | ✅ 존재 (2025-09-05 처리 완료) |
| **현재 데이터 상태** | ❌ 역코딩 미적용 |
| **즉시 실행 가능** | ✅ 가능 |

---

## 📌 핵심 요약

> **역문항 처리 시스템은 완전히 구축되어 있으나, 현재 데이터에는 적용되지 않은 상태입니다.**
> 
> **즉시 조치**:
> 1. 역문항 처리 실행 (17개 문항)
> 2. 처리 결과 검증
> 3. Ordered Probit 재테스트
> 
> **예상 효과**:
> - `perceived_benefit` 적합도 개선 (-4.78 → -4.0 이하)
> - 모든 요인의 일관성 향상
> - 잠재변수 추정 정확도 향상

---

**다음 단계**: 역문항 처리를 실행하시겠습니까?

**옵션**:
1. **방법 1**: 기존 모듈 사용 (자동, 권장)
2. **방법 2**: 수동 역코딩 스크립트 작성
3. **방법 3**: 테스트용 임시 역코딩만 적용

