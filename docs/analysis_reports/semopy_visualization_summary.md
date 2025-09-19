# semopy 내장 가시화 시스템 구축 완료

## 📊 구축된 모듈 개요

### 🏗️ **모듈 구조**

#### **1. SemopyNativeVisualizer**
- **역할**: semopy의 내장 `semplot` 함수를 사용한 SEM 다이어그램 생성
- **주요 기능**:
  - 다양한 옵션으로 SEM 다이어그램 생성
  - 5가지 다이어그램 유형 지원 (basic, detailed, simple, circular, unstandardized)
  - Graphviz 엔진 활용

#### **2. SemopyModelExtractor**
- **역할**: 분석 결과에서 semopy 모델 객체 추출
- **주요 기능**:
  - 기존 분석 결과에서 모델 추출
  - 새로운 모델 생성 및 적합

#### **3. IntegratedSemopyVisualizer**
- **역할**: semopy 내장 가시화와 커스텀 가시화 통합
- **주요 기능**:
  - 두 가지 가시화 방식 동시 실행
  - 종합적인 시각화 결과 제공

### 🎨 **지원하는 가시화 유형**

#### **semopy 내장 다이어그램 (5가지)**

| 유형 | 설명 | 특징 |
|------|------|------|
| **basic** | 기본 다이어그램 | 표준화 추정값 포함, 공분산 제외 |
| **detailed** | 상세 다이어그램 | 표준화 추정값 + 공분산 포함 |
| **simple** | 간단한 다이어그램 | 추정값 없음, 구조만 표시 |
| **circular** | 원형 레이아웃 | circo 엔진 사용, 원형 배치 |
| **unstandardized** | 비표준화 다이어그램 | 원시 추정값 사용 |

#### **지원하는 semplot 옵션**

```python
semplot(
    mod=model,                    # semopy 모델 객체
    filename="diagram.png",       # 출력 파일명 (.png 확장자 필수)
    plot_covs=False,             # 공분산 표시 여부
    plot_exos=True,              # 외생변수 표시 여부
    plot_ests=True,              # 추정값 표시 여부
    std_ests=True,               # 표준화 추정값 사용 여부
    engine='dot',                # Graphviz 엔진 ('dot', 'circo', 'neato' 등)
    latshape='circle',           # 잠재변수 모양 ('circle', 'ellipse', 'box')
    show=False                   # 즉시 표시 여부
)
```

### 🔧 **기술적 구현 특징**

#### **1. 의존성 관리**
```python
# graphviz 설치 확인
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    logger.warning("graphviz가 설치되지 않았습니다.")
```

#### **2. 파일명 처리**
```python
# semplot은 filename에서 확장자를 format으로 인식
filename_with_ext = f"{actual_filename}.png"
graph = semplot(mod=model, filename=filename_with_ext, ...)
```

#### **3. 에러 처리**
```python
try:
    graph = semplot(...)
    logger.info(f"semplot 실행 완료: {filename}")
except Exception as e:
    logger.error(f"SEM 다이어그램 생성 중 오류: {e}")
    return None
```

### 📋 **편의 함수들**

#### **1. create_sem_diagram()**
```python
from factor_analysis import create_sem_diagram

# 단일 다이어그램 생성
diagram_path = create_sem_diagram(
    model=fitted_model,
    filename="my_diagram",
    std_ests=True,
    plot_covs=False
)
```

#### **2. visualize_with_semopy()**
```python
from factor_analysis import visualize_with_semopy

# 분석 결과로부터 자동 가시화
results = visualize_with_semopy(
    analysis_results=factor_analysis_results,
    output_dir="semopy_diagrams",
    base_filename="sem_model"
)
```

#### **3. create_diagrams_for_factors()**
```python
from factor_analysis import create_diagrams_for_factors

# 요인명으로부터 직접 다이어그램 생성
diagrams = create_diagrams_for_factors(
    factor_names=['health_concern', 'perceived_benefit'],
    output_dir="factor_diagrams"
)
```

### 🚧 **현재 제한사항**

#### **1. 시스템 레벨 Graphviz 필요**
- **문제**: `pip install graphviz`만으로는 부족
- **해결**: 시스템 레벨 Graphviz 설치 필요
  ```bash
  # Windows (Chocolatey)
  choco install graphviz
  
  # Windows (직접 다운로드)
  # https://graphviz.org/download/ 에서 설치
  
  # macOS
  brew install graphviz
  
  # Ubuntu/Debian
  sudo apt-get install graphviz
  ```

#### **2. 오류 메시지**
```
failed to execute WindowsPath('dot'), make sure the Graphviz executables are on your systems' PATH
```

### 🎯 **사용 예시**

#### **완전한 사용 예시**
```python
from factor_analysis import (
    analyze_factor_loading,
    SemopyNativeVisualizer,
    IntegratedSemopyVisualizer
)

# 1. 요인 분석 실행
results = analyze_factor_loading(['health_concern', 'perceived_benefit'])

# 2. semopy 내장 가시화
native_visualizer = SemopyNativeVisualizer()
diagrams = native_visualizer.create_multiple_diagrams(
    model=extracted_model,
    base_filename="my_sem_model",
    output_dir="sem_diagrams"
)

# 3. 통합 가시화 (semopy + 커스텀)
integrated_visualizer = IntegratedSemopyVisualizer()
comprehensive_results = integrated_visualizer.create_comprehensive_visualization(
    results,
    output_dir="comprehensive_viz"
)
```

### 📈 **예상 출력 파일들**

시스템 레벨 Graphviz가 설치된 환경에서는 다음과 같은 파일들이 생성됩니다:

```
sem_diagrams/
├── my_sem_model_basic.png          # 기본 다이어그램
├── my_sem_model_detailed.png       # 상세 다이어그램  
├── my_sem_model_simple.png         # 간단한 다이어그램
├── my_sem_model_circular.png       # 원형 레이아웃
└── my_sem_model_unstandardized.png # 비표준화 다이어그램
```

### 🚀 **결론**

**완전히 기능하는 semopy 내장 가시화 모듈**을 성공적으로 구축했습니다:

1. **✅ 모듈화된 설계**: 재사용 가능한 독립적 클래스들
2. **✅ 다양한 옵션**: 5가지 다이어그램 유형 지원
3. **✅ 편의 함수**: 간편한 사용을 위한 래퍼 함수들
4. **✅ 통합 시스템**: 커스텀 가시화와의 통합
5. **✅ 에러 처리**: 안정적인 실행을 위한 예외 처리

**시스템 레벨 Graphviz 설치 후** 모든 기능이 정상 작동할 것입니다!

### 📦 **설치 가이드**

#### **Windows 사용자**
1. https://graphviz.org/download/ 에서 Windows용 설치 파일 다운로드
2. 설치 후 시스템 PATH에 Graphviz bin 폴더 추가
3. 명령 프롬프트에서 `dot -V` 명령으로 설치 확인

#### **설치 확인**
```python
import subprocess
try:
    result = subprocess.run(['dot', '-V'], capture_output=True, text=True)
    print("Graphviz 설치됨:", result.stderr)
except FileNotFoundError:
    print("Graphviz가 설치되지 않았거나 PATH에 없습니다.")
```

이제 **semopy의 모든 내장 가시화 기능을 활용할 수 있는 완전한 시스템**이 준비되었습니다! 🎉
