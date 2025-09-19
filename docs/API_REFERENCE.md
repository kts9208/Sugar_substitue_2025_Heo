# API Reference - Sugar Substitute Research

## 📚 목차

1. [분석 모듈](#분석-모듈)
2. [시각화 모듈](#시각화-모듈)
3. [유틸리티 모듈](#유틸리티-모듈)
4. [설정 관리](#설정-관리)
5. [데이터 구조](#데이터-구조)

## 🔬 분석 모듈

### Factor Analysis

#### `analyze_factor_loading(factor_name, config=None)`

요인분석을 수행합니다.

**Parameters:**
- `factor_name` (str or list): 분석할 요인명 또는 요인 리스트
- `config` (dict, optional): 분석 설정

**Returns:**
- `dict`: 분석 결과
  - `factor_loadings`: 요인적재량 DataFrame
  - `fit_indices`: 적합도 지수 dict
  - `model_info`: 모델 정보 dict

**Example:**
```python
from src.analysis.factor_analysis import analyze_factor_loading

# 단일 요인 분석
results = analyze_factor_loading("health_concern")

# 다중 요인 분석
results = analyze_factor_loading([
    "health_concern", 
    "perceived_benefit", 
    "purchase_intention"
])
```

#### `export_factor_results(results, output_dir="factor_analysis_results")`

요인분석 결과를 파일로 저장합니다.

**Parameters:**
- `results` (dict): 분석 결과
- `output_dir` (str): 출력 디렉토리

**Returns:**
- `dict`: 저장된 파일 경로들

### Path Analysis

#### `analyze_path_model(model_spec, variables, config=None)`

경로분석을 수행합니다.

**Parameters:**
- `model_spec` (str): 모델 스펙 (semopy 형식)
- `variables` (list): 분석 변수 리스트
- `config` (dict, optional): 분석 설정

**Returns:**
- `dict`: 분석 결과
  - `path_coefficients`: 경로계수 DataFrame
  - `fit_indices`: 적합도 지수 dict
  - `effects`: 직접/간접/총효과 dict

**Example:**
```python
from src.analysis.path_analysis import analyze_path_model, create_path_model

# 모델 스펙 생성
model_spec = create_path_model(
    model_type='simple_mediation',
    independent_var='health_concern',
    mediator_var='perceived_benefit',
    dependent_var='purchase_intention'
)

# 분석 실행
variables = ['health_concern', 'perceived_benefit', 'purchase_intention']
results = analyze_path_model(model_spec, variables)
```

#### `create_path_model(model_type, **kwargs)`

경로모델 스펙을 생성합니다.

**Parameters:**
- `model_type` (str): 모델 유형
  - `'simple_mediation'`: 단순 매개모델
  - `'multiple_mediation'`: 다중 매개모델
  - `'comprehensive'`: 포괄적 구조모델
  - `'saturated'`: 포화 모델
- `**kwargs`: 모델별 추가 파라미터

**Returns:**
- `str`: semopy 모델 스펙

### Reliability Analysis

#### `analyze_reliability(factor_data, factor_name)`

신뢰도 분석을 수행합니다.

**Parameters:**
- `factor_data` (DataFrame): 요인 데이터
- `factor_name` (str): 요인명

**Returns:**
- `dict`: 신뢰도 지표
  - `cronbach_alpha`: 크론바흐 알파
  - `composite_reliability`: 복합신뢰도
  - `ave`: 평균분산추출

## 🎨 시각화 모듈

### Correlation Visualizer

#### `create_correlation_heatmap(correlation_matrix, title="Correlation Matrix")`

상관관계 히트맵을 생성합니다.

**Parameters:**
- `correlation_matrix` (DataFrame): 상관관계 매트릭스
- `title` (str): 그래프 제목

**Returns:**
- `matplotlib.figure.Figure`: 생성된 그래프

### Discriminant Validity Analyzer

#### `analyze_discriminant_validity(factors_data, reliability_results)`

판별타당도를 분석합니다.

**Parameters:**
- `factors_data` (dict): 요인별 데이터
- `reliability_results` (dict): 신뢰도 분석 결과

**Returns:**
- `dict`: 판별타당도 결과
  - `fornell_larcker_matrix`: Fornell-Larcker 매트릭스
  - `htmt_matrix`: HTMT 매트릭스
  - `validity_status`: 타당도 상태

## 🛠️ 유틸리티 모듈

### Results Manager

#### `class ResultsManager(base_dir=".")`

결과 파일 버전 관리 클래스입니다.

**Methods:**

##### `save_results(analysis_type, results, files=None, auto_archive=True)`

분석 결과를 저장합니다.

**Parameters:**
- `analysis_type` (str): 분석 유형
- `results` (dict): 분석 결과
- `files` (dict, optional): 추가 파일들
- `auto_archive` (bool): 자동 아카이브 여부

**Returns:**
- `dict`: 저장된 파일 경로들

##### `archive_current_results(analysis_type, description="")`

현재 결과를 아카이브로 이동합니다.

**Parameters:**
- `analysis_type` (str): 분석 유형
- `description` (str): 아카이브 설명

**Returns:**
- `str`: 아카이브 디렉토리 경로

##### `get_latest_results(analysis_type)`

최신 결과 정보를 조회합니다.

**Parameters:**
- `analysis_type` (str): 분석 유형

**Returns:**
- `dict`: 최신 결과 정보

##### `list_versions(analysis_type)`

특정 분석 유형의 모든 버전을 조회합니다.

**Parameters:**
- `analysis_type` (str): 분석 유형

**Returns:**
- `list`: 버전 정보 리스트

**Example:**
```python
from src.utils.results_manager import ResultsManager

# 결과 관리자 생성
manager = ResultsManager()

# 결과 저장
saved_files = manager.save_results(
    "factor_analysis", 
    results, 
    auto_archive=True
)

# 버전 히스토리 확인
versions = manager.list_versions("factor_analysis")

# 이전 버전 복원
success = manager.restore_version("factor_analysis", "20250918_143022")
```

### 편의 함수들

#### `save_results(analysis_type, results, files=None, auto_archive=True)`

결과 저장 편의 함수입니다.

#### `archive_previous_results(analysis_type, description="")`

이전 결과 아카이브 편의 함수입니다.

#### `get_latest_results(analysis_type)`

최신 결과 조회 편의 함수입니다.

## ⚙️ 설정 관리

### Configuration

#### `config.py`

전역 설정을 관리하는 모듈입니다.

**주요 설정 그룹:**

##### `DATA_CONFIG`
데이터 디렉토리 경로 설정
```python
DATA_CONFIG = {
    "survey_data_dir": Path("data/processed/survey"),
    "config_dir": Path("data/config"),
    # ...
}
```

##### `ANALYSIS_CONFIG`
분석 관련 설정
```python
ANALYSIS_CONFIG = {
    "factor_analysis": {
        "min_loading_threshold": 0.5,
        "good_loading_threshold": 0.7,
        # ...
    }
}
```

##### `VISUALIZATION_CONFIG`
시각화 관련 설정
```python
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "korean_font": "Malgun Gothic"
}
```

#### 설정 함수들

##### `get_data_path(data_type="survey")`

데이터 경로를 반환합니다 (fallback 포함).

##### `ensure_directories()`

필요한 디렉토리들을 생성합니다.

## 📊 데이터 구조

### 입력 데이터 형식

#### 요인 데이터 (CSV)
```csv
q1,q2,q3,q4,q5
5,4,6,5,7
3,2,4,3,5
...
```

#### 설정 파일 (JSON)
```json
{
  "reverse_items": {
    "health_concern": ["q2", "q4"],
    "perceived_price": ["q1", "q3"]
  }
}
```

### 출력 데이터 형식

#### 분석 결과 (JSON)
```json
{
  "analysis_type": "factor_analysis",
  "timestamp": "20250918_143022",
  "model_info": {
    "n_observations": 500,
    "n_variables": 25
  },
  "factor_loadings": [...],
  "fit_indices": {
    "CFI": 0.95,
    "RMSEA": 0.06
  }
}
```

#### 요인적재량 (CSV)
```csv
Factor,Item,Loading,SE,Z,P
health_concern,q1,0.756,0.045,16.8,0.000
health_concern,q2,0.689,0.052,13.2,0.000
...
```

## 🔧 오류 처리

### 일반적인 예외

#### `DataNotFoundError`
데이터 파일을 찾을 수 없을 때 발생

#### `AnalysisError`
분석 실행 중 오류 발생 시

#### `ValidationError`
데이터 검증 실패 시

### 오류 처리 예시

```python
try:
    results = analyze_factor_loading("health_concern")
except DataNotFoundError as e:
    print(f"데이터 파일을 찾을 수 없습니다: {e}")
except AnalysisError as e:
    print(f"분석 실행 오류: {e}")
```

---

**Version**: 2.0  
**Last Updated**: 2025-09-18  
**Author**: Sugar Substitute Research Team
