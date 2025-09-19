# Sugar Substitute Research - 통합 분석 시스템

설탕 대체재에 대한 소비자 인식 및 구매의도 연구를 위한 통합 분석 시스템입니다.

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [시스템 구조](#시스템-구조)
- [설치 및 설정](#설치-및-설정)
- [사용법](#사용법)
- [분석 모듈](#분석-모듈)
- [결과 관리](#결과-관리)
- [문제 해결](#문제-해결)

## 🎯 프로젝트 개요

본 연구는 설탕 대체재에 대한 소비자의 인식과 구매의도를 분석하기 위해 다음과 같은 통계 분석을 수행합니다:

### 주요 분석 내용
- **요인분석 (Factor Analysis)**: 5개 주요 요인의 구조 확인
- **신뢰도 분석 (Reliability Analysis)**: Cronbach's α, CR, AVE 계산
- **판별타당도 검증 (Discriminant Validity)**: Fornell-Larcker 기준 적용
- **상관관계 분석 (Correlation Analysis)**: 요인 간 상관관계 분석
- **경로분석 (Path Analysis)**: 구조방정식 모델링 (SEM)
- **조절효과 분석 (Moderation Analysis)**: 조절변수 효과 검증
- **다항로짓 분석 (Multinomial Logit)**: 선택 모델링

### 연구 요인
1. **건강관심도 (Health Concern)**
2. **지각된 유익성 (Perceived Benefit)**
3. **구매의도 (Purchase Intention)**
4. **지각된 가격 (Perceived Price)**
5. **영양지식 (Nutrition Knowledge)**

## 🏗️ 시스템 구조

```
Sugar_substitue_2025_Heo/
├── main.py                     # 통합 실행 스크립트
├── config.py                   # 전역 설정 파일
├── README.md                   # 프로젝트 문서
│
├── data/                       # 데이터 디렉토리
│   ├── raw/                    # 원본 데이터
│   ├── processed/              # 전처리된 데이터
│   │   ├── survey/            # 설문조사 데이터
│   │   └── dce/               # DCE 데이터
│   └── config/                # 설정 파일들
│
├── src/                        # 소스 코드
│   ├── analysis/              # 분석 모듈들
│   │   ├── factor_analysis/   # 요인분석
│   │   ├── moderation_analysis/ # 조절효과 분석
│   │   ├── multinomial_logit/ # 다항로짓 분석
│   │   └── utility_function/  # 효용함수 분석
│   ├── visualization/         # 시각화 모듈들
│   └── utils/                 # 유틸리티 모듈들
│
├── scripts/                    # 실행 스크립트들
│   ├── run_factor_analysis.py
│   ├── run_reliability_analysis.py
│   ├── run_path_analysis.py
│   ├── run_complete_analysis.py
│   └── manage_results.py
│
├── results/                    # 분석 결과
│   ├── current/               # 최신 결과
│   │   ├── factor_analysis/
│   │   ├── path_analysis/
│   │   ├── reliability_analysis/
│   │   └── ...
│   └── archive/               # 아카이브된 결과
│       ├── 2025-09-18/
│       └── historical/
│
├── tests/                      # 테스트 파일들
├── docs/                       # 문서화
│   └── analysis_reports/      # 분석 보고서들
├── notebooks/                  # Jupyter 노트북들
└── logs/                       # 로그 파일들
```

## 🚀 설치 및 설정

### 1. 필요한 패키지 설치

```bash
pip install pandas numpy scipy semopy matplotlib seaborn pathlib
```

### 2. 디렉토리 구조 생성

```bash
python config.py
```

### 3. 데이터 준비

설문조사 데이터를 `data/processed/survey/` 디렉토리에 배치:
- `health_concern.csv`
- `perceived_benefit.csv`
- `purchase_intention.csv`
- `perceived_price.csv`
- `nutrition_knowledge.csv`

## 📖 사용법

### 1. 통합 실행 (권장)

```bash
# 대화형 메뉴 실행
python main.py

# 전체 분석 파이프라인 실행
python main.py --all

# 개별 분석 실행
python main.py --factor          # 요인분석만
python main.py --reliability     # 신뢰도 분석만
python main.py --path           # 경로분석만

# 결과 요약 보기
python main.py --results
```

### 2. 개별 스크립트 실행

```bash
# 요인분석
python scripts/run_factor_analysis.py --all

# 신뢰도 분석
python scripts/run_reliability_analysis.py

# 경로분석
python scripts/run_path_analysis.py --model comprehensive

# 전체 분석 파이프라인
python scripts/run_complete_analysis.py --core-only
```

### 3. 결과 관리

```bash
# 현재 결과 상태 확인
python scripts/manage_results.py --status

# 특정 분석의 버전 히스토리 확인
python scripts/manage_results.py --history factor_analysis

# 결과 아카이브
python scripts/manage_results.py --archive path_analysis --description "새로운 모델"

# 이전 버전 복원
python scripts/manage_results.py --restore factor_analysis 20250918_143022

# 오래된 버전 정리
python scripts/manage_results.py --cleanup factor_analysis --keep 3
```

## 🔬 분석 모듈

### 1. 요인분석 (Factor Analysis)
- **목적**: 5개 요인의 구조 확인
- **방법**: 확인적 요인분석 (CFA)
- **출력**: Factor loadings, 적합도 지수
- **위치**: `src/analysis/factor_analysis/`

### 2. 신뢰도 분석 (Reliability Analysis)
- **목적**: 측정도구의 신뢰도 검증
- **지표**: Cronbach's α, CR, AVE
- **기준**: α ≥ 0.7, CR ≥ 0.7, AVE ≥ 0.5
- **출력**: 신뢰도 지표 표, 시각화

### 3. 경로분석 (Path Analysis)
- **목적**: 요인 간 인과관계 분석
- **방법**: 구조방정식 모델링 (SEM)
- **모델**: 단순매개, 다중매개, 포괄적 구조모델
- **출력**: 경로계수, 매개효과, 적합도 지수

### 4. 판별타당도 검증 (Discriminant Validity)
- **목적**: 요인 간 구별되는 정도 확인
- **방법**: Fornell-Larcker 기준, HTMT
- **기준**: √AVE > 상관계수, HTMT < 0.85
- **출력**: 판별타당도 매트릭스

### 5. 조절효과 분석 (Moderation Analysis)
- **목적**: 조절변수의 효과 검증
- **방법**: 상호작용 효과 분석
- **출력**: 조절효과 계수, 시각화

## 📊 결과 관리

### 버전 관리 시스템
- **현재 결과**: `results/current/` - 가장 최신 분석 결과
- **아카이브**: `results/archive/` - 이전 버전들의 백업
- **메타데이터**: `results/metadata.json` - 버전 정보 추적

### 자동 아카이브
- 새로운 분석 실행 시 기존 결과 자동 아카이브
- 타임스탬프 기반 버전 관리
- 최대 보관 버전 수 설정 가능

### 결과 파일 구조
```
results/current/factor_analysis/
├── results_20250918_143022.json      # 메인 결과
├── factor_loadings_20250918_143022.csv
├── fit_indices_20250918_143022.csv
└── visualization_20250918_143022.png
```

## 🔧 문제 해결

### 자주 발생하는 문제

#### 1. 모듈 임포트 오류
```bash
ModuleNotFoundError: No module named 'src.analysis'
```
**해결책**: Python 경로 설정 확인
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src"
```

#### 2. 데이터 파일을 찾을 수 없음
```bash
❌ 설문조사 데이터 디렉토리를 찾을 수 없습니다.
```
**해결책**: 데이터 파일 위치 확인
- `data/processed/survey/` 또는 `processed_data/survey_data/`에 CSV 파일들이 있는지 확인

#### 3. 분석 실행 실패
```bash
❌ 요인분석 실패
```
**해결책**: 로그 파일 확인
```bash
tail -f logs/factor_analysis.log
```

#### 4. 결과 저장 실패
**해결책**: 디렉토리 권한 및 용량 확인
```bash
python config.py  # 디렉토리 재생성
```

### 로그 파일 위치
- **메인 로그**: `logs/main_analysis.log`
- **요인분석 로그**: `logs/factor_analysis.log`
- **경로분석 로그**: `logs/path_analysis.log`
- **신뢰도 분석 로그**: `logs/reliability_analysis.log`

### 지원 및 문의
- **이슈 리포팅**: GitHub Issues
- **문서**: `docs/analysis_reports/`
- **예제**: `notebooks/`

## 📈 성능 최적화

### 권장 실행 순서
1. **요인분석** → 2. **신뢰도 분석** → 3. **경로분석** → 4. **판별타당도**

### 메모리 사용량 최적화
- 대용량 데이터셋의 경우 청크 단위 처리
- 불필요한 중간 결과 정리
- 분석별 독립 실행 권장

### 실행 시간 단축
- 부트스트래핑 샘플 수 조정 (`config.py`에서 설정)
- 병렬 처리 활용
- 캐시된 결과 재사용

---

**Version**: 2.0 (Reorganized)  
**Last Updated**: 2025-09-18  
**Author**: Sugar Substitute Research Team
