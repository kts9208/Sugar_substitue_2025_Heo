# 🏗️ Sugar Substitute Research - 통합시스템 아키텍처 가이드

## 📋 시스템 개요

Sugar Substitute Research 통합시스템은 **5계층 아키텍처**로 구성되어 있으며, 설탕 대체재 구매의도 연구를 위한 포괄적인 분석 환경을 제공합니다.

## 🎯 1. User Interface Layer (사용자 인터페이스 계층)

### 📄 main.py - 통합 실행 시스템
**역할**: 시스템의 단일 진입점 (Single Entry Point)
```python
# 주요 기능
- 대화형 메뉴 인터페이스
- 명령행 옵션 처리 (--factor, --reliability, --path, --all)
- 사전 요구사항 검증
- 분석 파이프라인 조율
- 오류 처리 및 로깅
```

**사용법**:
```bash
python main.py                    # 대화형 메뉴
python main.py --factor           # 요인분석
python main.py --reliability      # 신뢰도 분석
python main.py --path             # 경로분석
python main.py --all              # 전체 분석
```

### ⚙️ config.py - 설정 관리
**역할**: 중앙화된 설정 및 경로 관리
```python
# 주요 기능
- 데이터 경로 설정 (get_data_path)
- 결과 디렉토리 관리 (get_results_path)
- 설정 파일 경로 (get_config_file)
- 디렉토리 자동 생성 (ensure_directories)
- Fallback 경로 지원
```

### 📚 README.md - 프로젝트 문서
**역할**: 프로젝트 개요 및 시작 가이드

## 🚀 2. Execution Scripts Layer (실행 스크립트 계층)

### 🔍 scripts/run_factor_analysis.py
**역할**: 요인분석 전용 실행기
```python
# 주요 기능
- 개별 요인 분석 (--factor health_concern)
- 전체 요인 분석 (--all)
- 확인적 요인분석 (CFA)
- 요인적재량 계산
- 모델 적합도 평가
```

**분석 대상**:
- 건강관심도 (health_concern): 7개 문항
- 지각된유익성 (perceived_benefit): 7개 문항  
- 구매의도 (purchase_intention): 4개 문항
- 지각된가격 (perceived_price): 4개 문항
- 영양지식 (nutrition_knowledge): 21개 문항

### 📊 scripts/run_reliability_analysis.py
**역할**: 신뢰도 분석 전용 실행기
```python
# 주요 기능
- Cronbach's Alpha 계산
- 복합신뢰도 (CR) 계산
- 평균분산추출 (AVE) 계산
- 문항-총점 상관관계
- 문항간 상관관계 분석
```

### 🔗 scripts/run_path_analysis.py
**역할**: 경로분석 및 구조방정식 모델링
```python
# 주요 기능
- 단순 매개분석 (simple_mediation)
- 다중 매개분석 (multiple_mediation)
- 포괄적 구조모델 (comprehensive_structural)
- 포화모델 (saturated)
- 직접효과/간접효과 계산
```

### 🎯 scripts/run_complete_analysis.py
**역할**: 전체 분석 파이프라인 조율
```python
# 분석 순서
1. 요인분석 → 측정모델 검증
2. 신뢰도 분석 → 측정도구 검증
3. 경로분석 → 구조모델 검증
4. 결과 통합 → 종합 보고서 생성
```

### 📁 scripts/manage_results.py
**역할**: 결과 파일 관리 및 버전 추적
```python
# 주요 기능
- 결과 상태 확인 (--status)
- 결과 아카이브 (--archive)
- 버전 히스토리 (--history)
- 메타데이터 관리
- 자동 백업
```

## 🔬 3. Core Analysis Modules (핵심 분석 모듈 계층)

### 📈 src/analysis/factor_analysis/
**실제 구성 요소**:
```
factor_analysis/
├── __init__.py
├── data_loader.py              # 데이터 로딩 및 전처리
├── factor_analyzer.py          # 요인분석 핵심 엔진
├── reliability_calculator.py   # 신뢰도 계산기
├── reliability_visualizer.py   # 신뢰도 시각화
├── comparison_analyzer.py      # 요인 비교 분석
├── semopy_correlations.py      # semopy 기반 상관분석
├── semopy_native_visualizer.py # semopy 네이티브 시각화
├── visualizer.py               # 일반 시각화
├── results_exporter.py         # 결과 내보내기
├── config.py                   # 요인분석 설정
├── example_usage.py            # 사용 예시
├── test_factor_analysis.py     # 테스트 코드
└── README.md                   # 모듈 문서
```

**핵심 기능**:
- **semopy 기반 확인적 요인분석**: 구조방정식 모델링
- **요인적재량 계산**: 각 문항의 요인에 대한 기여도
- **모델 적합도 평가**: CFI, TLI, RMSEA, SRMR 지수
- **신뢰도 분석**: Cronbach's Alpha, CR, AVE
- **요인 간 상관관계**: 판별타당도 검증
- **시각화**: 요인적재량 차트, 신뢰도 대시보드

### 🔗 src/analysis/path_analysis/
**역할**: 구조방정식 모델링 및 매개분석
```
path_analysis/
├── __init__.py
├── mediation_analyzer.py       # 매개분석 (단순/다중)
├── structural_model.py         # 구조방정식 모델
├── effect_calculator.py        # 직접/간접 효과 계산
├── bootstrap_analysis.py       # 부트스트랩 신뢰구간
└── path_visualizer.py          # 경로도표 생성
```

**핵심 기능**:
- **매개분석**: 건강관심도 → 지각된유익성 → 구매의도
- **조절효과**: 지각된가격의 조절역할 분석
- **직접/간접 효과**: 경로계수 및 매개효과 계산
- **부트스트랩**: 신뢰구간 및 유의성 검증

### 🎯 src/analysis/moderation_analysis/
**역할**: 조절효과 분석
- 지각된가격의 조절역할
- 영양지식의 조절효과
- 상호작용 효과 분석

### 📊 src/analysis/multinomial_logit/
**역할**: 다항로짓 분석 (DCE 데이터)
- 선택 확률 모델링
- 효용함수 추정
- 선호도 분석

### 🔧 src/analysis/utility_function/
**역할**: 효용함수 분석
- 개별 효용 계산
- 집단 효용 추정
- 선호 이질성 분석

### 🎨 src/visualization/
**구성 요소**:
```
visualization/
├── __init__.py
├── factor_plots.py         # 요인분석 시각화
├── path_diagrams.py        # 경로도표
├── reliability_charts.py   # 신뢰도 차트
└── dashboard.py            # 통합 대시보드
```

### 🛠️ src/utils/
**구성 요소**:
```
utils/
├── __init__.py
├── results_manager.py      # 결과 관리자
├── data_validator.py       # 데이터 검증
├── logger.py               # 로깅 유틸리티
└── file_handler.py         # 파일 처리
```

## 📊 4. Data Layer (데이터 계층)

### 📁 data/raw/
**역할**: 원본 데이터 저장소
- 수집된 원시 데이터
- 변경 불가능한 원본 보존

### 📁 data/processed/survey/
**역할**: 전처리된 설문 데이터
```
survey/
├── health_concern.csv      # 건강관심도 (300×7)
├── perceived_benefit.csv   # 지각된유익성 (300×7)
├── purchase_intention.csv  # 구매의도 (300×4)
├── perceived_price.csv     # 지각된가격 (300×4)
├── nutrition_knowledge.csv # 영양지식 (300×21)
└── factors_summary.csv     # 요인 요약
```

### 📁 data/config/
**역할**: 설정 파일 저장
- reverse_items_config.json: 역코딩 문항 설정
- analysis_config.json: 분석 매개변수

### 📁 Raw data/
**역할**: 원본 데이터 보관소 (백업)

## 📁 5. Output & Documentation Layer (결과 및 문서 계층)

### 📊 results/current/
**역할**: 현재 분석 결과 저장
```
current/
├── factor_analysis/        # 요인분석 결과
├── reliability/            # 신뢰도 분석 결과
├── path_analysis/          # 경로분석 결과
├── visualizations/         # 시각화 결과
└── comprehensive/          # 통합 분석 결과
```

### 📦 results/archive/
**역할**: 과거 결과 아카이브
- 타임스탬프 기반 버전 관리
- 자동 아카이브 시스템
- 메타데이터 추적

### 📚 docs/
**구성 요소**:
```
docs/
├── USER_GUIDE.md           # 사용자 가이드
├── API_REFERENCE.md        # API 문서
├── CHANGELOG.md            # 변경 로그
└── analysis_reports/       # 분석 보고서
```

### 📋 logs/
**역할**: 실행 로그 관리
- main_analysis.log: 주요 분석 로그
- error.log: 오류 로그
- performance.log: 성능 로그

### 🧪 tests/
**역할**: 테스트 코드 저장
- 단위 테스트
- 통합 테스트
- 성능 테스트

### 📓 notebooks/
**역할**: 분석 노트북 저장
- 탐색적 데이터 분석
- 프로토타입 개발
- 결과 해석 및 시각화

## 🔄 데이터 플로우

```
1. 데이터 입력: Raw data/ → data/processed/survey/
2. 분석 실행: main.py → scripts/ → src/analysis/
3. 결과 생성: src/analysis/ → results/current/
4. 시각화: src/visualization/ → results/current/visualizations/
5. 아카이브: results/current/ → results/archive/
```

## 🎯 주요 특징

### ✅ 모듈화 설계
- 각 기능별 독립적인 모듈
- 재사용 가능한 컴포넌트
- 확장 가능한 구조

### ✅ 자동화된 워크플로우
- 단일 명령으로 전체 분석 실행
- 자동 결과 저장 및 아카이브
- 오류 처리 및 복구

### ✅ 버전 관리
- 타임스탬프 기반 파일 명명
- 자동 아카이브 시스템
- 메타데이터 추적

### ✅ 사용자 친화적 인터페이스
- 대화형 메뉴 시스템
- 명령행 옵션 지원
- 실시간 진행 상황 표시

이 통합시스템을 통해 설탕 대체재 구매의도 연구를 체계적이고 효율적으로 수행할 수 있습니다.
