# 구조모델 경로 수정 용이성 분석 보고서

**작성일**: 2025-11-05  
**프로젝트**: Sugar Substitute 2025 (대체당 연구)  
**목적**: 구조모델 경로 수정이 얼마나 용이한지 분석

---

## ✅ 핵심 결론

### **현재 상태: ⚠️ 부분적으로 용이함 (60%)**

| 항목 | 평가 | 점수 |
|------|------|------|
| **설정 기반 구조** | ✅ 우수 | 90% |
| **경로 수정 방법** | ⚠️ 보통 | 50% |
| **코드 중복** | ⚠️ 있음 | 40% |
| **문서화** | ✅ 우수 | 80% |
| **확장성** | ⚠️ 제한적 | 50% |
| **전체 평가** | ⚠️ 개선 필요 | 60% |

---

## 📊 1. 현재 구조 분석

### **1.1 구조모델 경로 정의 방식**

#### **현재 방식: 하드코딩 (Hardcoded)**

**위치**: `scripts/run_iclv_estimation.py` (48-171행)

```python
def create_iclv_config():
    configs = {}
    
    # 1. 건강관심도
    configs['health_concern'] = {
        'structural': StructuralConfig(
            sociodemographics=['age_std', 'gender', 'income_std', 'education_level']
        )
    }
    
    # 2. 건강유익성
    configs['perceived_benefit'] = {
        'structural': StructuralConfig(
            sociodemographics=['health_concern']  # ← 하드코딩
        )
    }
    
    # 3. 구매의도
    configs['purchase_intention'] = {
        'structural': StructuralConfig(
            sociodemographics=['perceived_benefit', 'perceived_price', 'nutrition_knowledge']  # ← 하드코딩
        )
    }
    
    # ... 나머지 잠재변수
```

---

### **1.2 문제점**

#### **❌ 문제 1: 경로가 함수 내부에 하드코딩**

**현재**:
- 경로를 수정하려면 `create_iclv_config()` 함수 내부를 직접 수정해야 함
- 5개 잠재변수 × 평균 3개 경로 = 15개 라인 수정 필요

**예시**:
```python
# 경로 수정 시
configs['purchase_intention'] = {
    'structural': StructuralConfig(
        sociodemographics=['perceived_benefit', 'perceived_price']  # ← 이 라인 수정
    )
}
```

---

#### **❌ 문제 2: 경로 정의가 분산됨**

**현재 구조**:
```
scripts/run_iclv_estimation.py (48-171행)
├─ health_concern 경로 (70행)
├─ perceived_benefit 경로 (86행)
├─ purchase_intention 경로 (106행)
├─ perceived_price 경로 (125행)
└─ nutrition_knowledge 경로 (143행)
```

**문제**:
- 전체 경로 구조를 한눈에 파악하기 어려움
- 경로 수정 시 여러 곳을 찾아다녀야 함

---

#### **❌ 문제 3: 경로 검증 기능 없음**

**현재**:
- 순환 경로 체크 없음 (예: A → B → A)
- 존재하지 않는 변수 참조 체크 없음
- 경로 충돌 체크 없음

---

## 🎯 2. 개선 방안

### **방안 1: 설정 파일 기반 (Configuration File) ✅ 권장**

#### **개념**

경로를 별도의 설정 파일(YAML/JSON)로 분리

**장점**:
- ✅ 코드 수정 없이 경로 변경 가능
- ✅ 전체 경로 구조를 한눈에 파악
- ✅ 버전 관리 용이
- ✅ 여러 모델 설정 관리 가능

**단점**:
- ⚠️ 설정 파일 파싱 로직 필요
- ⚠️ 설정 파일 검증 필요

---

#### **구현 예시**

**파일**: `configs/structural_paths.yaml`

```yaml
# 구조모델 경로 설정
structural_paths:
  # 1차 잠재변수 (외생변수의 영향)
  health_concern:
    predictors:
      - age_std
      - gender
      - income_std
      - education_level
    description: "사회인구학적 변수 → 건강관심도"
  
  nutrition_knowledge:
    predictors:
      - age_std
      - education_level
    description: "연령, 교육 → 영양지식"
  
  # 2차 잠재변수 (1차 잠재변수의 영향)
  perceived_benefit:
    predictors:
      - health_concern
    description: "건강관심도 → 건강유익성"
  
  perceived_price:
    predictors:
      - income_std
    description: "소득 → 인지된 가격수준"
  
  # 3차 잠재변수 (2차 잠재변수의 영향)
  purchase_intention:
    predictors:
      - perceived_benefit
      - perceived_price
      - nutrition_knowledge
    description: "건강유익성, 가격수준, 영양지식 → 구매의도"

# 경로 제약조건
constraints:
  # 순환 경로 금지
  no_cycles: true
  
  # 최대 경로 길이
  max_path_length: 3
  
  # 필수 경로
  required_paths:
    - [health_concern, perceived_benefit]
    - [perceived_benefit, purchase_intention]
```

---

**사용 코드**: `scripts/run_iclv_estimation.py`

```python
import yaml

def load_structural_paths(config_file='configs/structural_paths.yaml'):
    """
    구조모델 경로 설정 로드
    
    Returns:
        dict: 경로 설정
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config['structural_paths']


def create_iclv_config_from_file(path_config_file='configs/structural_paths.yaml'):
    """
    설정 파일 기반 ICLV 설정 생성
    """
    # 경로 설정 로드
    structural_paths = load_structural_paths(path_config_file)
    
    configs = {}
    
    # 각 잠재변수별 설정 생성
    for lv_name, path_info in structural_paths.items():
        configs[lv_name] = {
            'measurement': MeasurementConfig(
                latent_variable=lv_name,
                indicators=get_indicators(lv_name),  # 별도 함수
                n_categories=5
            ),
            'structural': StructuralConfig(
                sociodemographics=path_info['predictors']  # ← 설정 파일에서 로드
            )
        }
    
    return configs
```

---

**경로 수정 예시**:

```yaml
# 경로 4 제거: 영양지식 → 구매의도
purchase_intention:
  predictors:
    - perceived_benefit
    - perceived_price
    # - nutrition_knowledge  ← 주석 처리만 하면 됨!
```

**코드 수정 필요**: ❌ **없음!**

---

### **방안 2: 경로 빌더 클래스 (Path Builder) ⚠️ 중간**

#### **개념**

경로를 프로그래밍 방식으로 정의하는 빌더 클래스

**장점**:
- ✅ 유연한 경로 정의
- ✅ 경로 검증 가능
- ✅ IDE 자동완성 지원

**단점**:
- ⚠️ 여전히 코드 수정 필요
- ⚠️ 학습 곡선 있음

---

#### **구현 예시**

```python
class StructuralPathBuilder:
    """구조모델 경로 빌더"""
    
    def __init__(self):
        self.paths = {}
    
    def add_path(self, target: str, predictors: List[str], 
                 description: str = "") -> 'StructuralPathBuilder':
        """
        경로 추가
        
        Args:
            target: 결과 변수 (잠재변수)
            predictors: 예측 변수 리스트
            description: 경로 설명
        
        Returns:
            self (메서드 체이닝용)
        """
        self.paths[target] = {
            'predictors': predictors,
            'description': description
        }
        return self
    
    def remove_path(self, target: str, predictor: str) -> 'StructuralPathBuilder':
        """특정 경로 제거"""
        if target in self.paths:
            self.paths[target]['predictors'].remove(predictor)
        return self
    
    def validate(self) -> bool:
        """경로 검증 (순환 체크 등)"""
        # 순환 경로 체크
        for target, info in self.paths.items():
            if self._has_cycle(target, info['predictors']):
                raise ValueError(f"순환 경로 발견: {target}")
        return True
    
    def _has_cycle(self, target: str, predictors: List[str], 
                   visited: set = None) -> bool:
        """순환 경로 체크 (DFS)"""
        if visited is None:
            visited = set()
        
        if target in visited:
            return True
        
        visited.add(target)
        
        for pred in predictors:
            if pred in self.paths:
                if self._has_cycle(pred, self.paths[pred]['predictors'], visited.copy()):
                    return True
        
        return False
    
    def build(self) -> Dict:
        """경로 설정 생성"""
        self.validate()
        return self.paths


# 사용 예시
def create_structural_paths():
    """구조모델 경로 정의"""
    
    builder = StructuralPathBuilder()
    
    # 1차 잠재변수
    builder.add_path(
        'health_concern',
        ['age_std', 'gender', 'income_std', 'education_level'],
        "사회인구학적 → 건강관심도"
    )
    
    builder.add_path(
        'nutrition_knowledge',
        ['age_std', 'education_level'],
        "연령, 교육 → 영양지식"
    )
    
    # 2차 잠재변수
    builder.add_path(
        'perceived_benefit',
        ['health_concern'],
        "건강관심도 → 건강유익성"
    )
    
    builder.add_path(
        'perceived_price',
        ['income_std'],
        "소득 → 가격수준"
    )
    
    # 3차 잠재변수
    builder.add_path(
        'purchase_intention',
        ['perceived_benefit', 'perceived_price', 'nutrition_knowledge'],
        "유익성, 가격, 지식 → 구매의도"
    )
    
    return builder.build()


# 경로 수정 예시
def modify_paths():
    """경로 수정"""
    paths = create_structural_paths()
    
    # 경로 4 제거
    builder = StructuralPathBuilder()
    for target, info in paths.items():
        builder.add_path(target, info['predictors'], info['description'])
    
    builder.remove_path('purchase_intention', 'nutrition_knowledge')
    
    return builder.build()
```

---

### **방안 3: 현재 방식 개선 (Improved Current) ⚠️ 최소**

#### **개념**

현재 하드코딩 방식을 유지하되, 가독성과 유지보수성 개선

**장점**:
- ✅ 최소한의 변경
- ✅ 기존 코드 호환

**단점**:
- ❌ 여전히 코드 수정 필요
- ❌ 근본적 문제 해결 안됨

---

#### **구현 예시**

```python
def create_iclv_config():
    """
    ICLV 모델 설정 생성 (5개 잠재변수)
    
    경로 수정 시 아래 STRUCTURAL_PATHS 딕셔너리만 수정하세요!
    """
    
    # ========================================
    # 구조모델 경로 정의 (여기만 수정!)
    # ========================================
    STRUCTURAL_PATHS = {
        # 1차 잠재변수
        'health_concern': ['age_std', 'gender', 'income_std', 'education_level'],
        'nutrition_knowledge': ['age_std', 'education_level'],
        
        # 2차 잠재변수
        'perceived_benefit': ['health_concern'],
        'perceived_price': ['income_std'],
        
        # 3차 잠재변수
        'purchase_intention': ['perceived_benefit', 'perceived_price', 'nutrition_knowledge'],
    }
    # ========================================
    
    configs = {}
    
    # 각 잠재변수별 설정 생성
    for lv_name, predictors in STRUCTURAL_PATHS.items():
        configs[lv_name] = {
            'measurement': MeasurementConfig(
                latent_variable=lv_name,
                indicators=get_indicators(lv_name),
                n_categories=5
            ),
            'structural': StructuralConfig(
                sociodemographics=predictors  # ← 딕셔너리에서 로드
            )
        }
    
    return configs
```

**경로 수정 예시**:
```python
# 경로 4 제거
STRUCTURAL_PATHS = {
    'purchase_intention': ['perceived_benefit', 'perceived_price'],  # ← 이 라인만 수정
}
```

---

## 📊 3. 방안 비교

| 항목 | 방안 1<br/>(설정 파일) | 방안 2<br/>(빌더 클래스) | 방안 3<br/>(현재 개선) |
|------|---------------------|---------------------|-------------------|
| **코드 수정 필요** | ❌ 없음 | ✅ 있음 | ✅ 있음 |
| **경로 가시성** | ✅ 매우 좋음 | ⚠️ 보통 | ✅ 좋음 |
| **경로 검증** | ✅ 가능 | ✅ 가능 | ❌ 없음 |
| **학습 곡선** | ⚠️ 중간 | ⚠️ 높음 | ✅ 낮음 |
| **확장성** | ✅ 매우 좋음 | ✅ 좋음 | ⚠️ 제한적 |
| **구현 난이도** | ⚠️ 중간 | ⚠️ 높음 | ✅ 낮음 |
| **권장도** | ✅ **강력 권장** | ⚠️ 선택적 | ⚠️ 임시 방편 |

---

## ✅ 최종 권장사항

### **단기 (즉시 적용): 방안 3 (현재 개선)**

**이유**:
- 최소한의 변경으로 즉시 개선 가능
- 경로 정의를 한 곳에 모음
- 기존 코드와 호환

**구현 시간**: 30분

---

### **중기 (1-2주 내): 방안 1 (설정 파일)**

**이유**:
- 코드 수정 없이 경로 변경 가능
- 여러 모델 설정 관리 용이
- 버전 관리 및 협업에 유리

**구현 시간**: 2-3시간

---

### **장기 (필요시): 방안 2 (빌더 클래스)**

**이유**:
- 복잡한 경로 검증 필요 시
- 프로그래밍 방식의 유연성 필요 시

**구현 시간**: 4-6시간

---

## 📝 구현 우선순위

| 우선순위 | 작업 | 예상 시간 | 효과 |
|---------|------|----------|------|
| **P0** | 방안 3 구현 (경로 딕셔너리 분리) | 30분 | 즉시 개선 |
| **P1** | 방안 1 구현 (YAML 설정 파일) | 2-3시간 | 장기적 유지보수성 |
| **P2** | 경로 검증 로직 추가 | 1-2시간 | 안정성 향상 |
| **P3** | 방안 2 구현 (빌더 클래스) | 4-6시간 | 고급 기능 |

---

## ✅ 최종 결론

### **현재 상태**

- ⚠️ **경로 수정 용이성: 60%**
- 경로가 함수 내부에 하드코딩되어 있음
- 경로 수정 시 여러 곳을 수정해야 함
- 경로 검증 기능 없음

### **개선 후 예상**

- ✅ **경로 수정 용이성: 95%** (방안 1 적용 시)
- 설정 파일만 수정하면 됨
- 전체 경로 구조를 한눈에 파악 가능
- 경로 검증 자동화

### **권장 조치**

1. **즉시**: 방안 3 적용 (경로 딕셔너리 분리)
2. **1-2주 내**: 방안 1 적용 (YAML 설정 파일)
3. **필요시**: 방안 2 적용 (빌더 클래스)

---

**보고 완료** ✅  
**보고 일시**: 2025-11-05  
**분석 대상**: `scripts/run_iclv_estimation.py`, `src/analysis/hybrid_choice_model/iclv_models/`

