# 🧹 Sugar Substitute Research - 최종 정리 가이드

## 📊 현재 상태 분석

새로운 구조가 성공적으로 구축되었지만, 아직 정리가 필요한 레거시 디렉토리들이 남아있습니다.

### ✅ 새로운 구조 (유지)
```
Sugar_substitue_2025_Heo/
├── main.py                 # 통합 실행 시스템
├── config.py               # 설정 관리  
├── README.md               # 프로젝트 문서
├── Raw data/               # 원본 데이터 (중요 - 보존)
├── data/                   # 새로운 데이터 구조
├── src/                    # 모듈화된 소스 코드
├── scripts/                # 통합 실행 스크립트
├── results/                # 결과 관리 시스템
├── docs/                   # 문서화
├── logs/                   # 로그 관리
├── tests/                  # 테스트 코드
└── notebooks/              # 분석 노트북
```

### 🗑️ 정리 대상 (레거시 디렉토리)

다음 디렉토리들은 이미 새로운 구조로 통합되었으므로 안전하게 삭제할 수 있습니다:

#### 1. 분석 결과 디렉토리 (8개)
```
comprehensive_mediation_results/     # → results/archive/로 이동됨
moderation_analysis_results/         # → results/archive/로 이동됨  
path_analysis_effects_test_results/  # → results/archive/로 이동됨
real_data_effects_test_results/      # → results/archive/로 이동됨
rpl_analysis_results/                # → results/archive/로 이동됨
simple_effects_test_results/         # → results/archive/로 이동됨
utility_function_analysis_results/   # → results/archive/로 이동됨
utility_function_results/            # → results/archive/로 이동됨
```

#### 2. 기타 디렉토리 (3개)
```
final_english_charts/                # → results/current/에 통합됨
sem_individual_implementation/       # → src/analysis/에 통합됨
test_diagram/                        # → 임시 테스트 파일
```

#### 3. processed_data 하위 디렉토리 (7개)
```
processed_data/dce_data/             # → data/processed/dce/로 이동됨
processed_data/docs/                 # → docs/로 이동됨
processed_data/examples/             # → notebooks/로 이동됨
processed_data/modules/              # → src/로 이동됨
processed_data/survey_data/          # → data/processed/survey/로 이동됨
processed_data/survey_data_backup_restore/  # → 백업 완료
processed_data/tests/                # → tests/로 이동됨
```

#### 4. 캐시 파일
```
__pycache__/                         # Python 캐시
scripts/__pycache__/                 # Python 캐시
src/__pycache__/                     # Python 캐시
```

## 🛡️ 안전한 정리 절차

### 1단계: 중요 결과 확인
정리 전에 다음 위치에 중요한 결과가 백업되어 있는지 확인:
```bash
ls -la results/archive/
ls -la final_english_charts/
```

### 2단계: 수동 정리 (권장)
Windows 탐색기나 명령어로 다음 디렉토리들을 삭제:

```bash
# 분석 결과 디렉토리 삭제
rmdir /s comprehensive_mediation_results
rmdir /s moderation_analysis_results  
rmdir /s path_analysis_effects_test_results
rmdir /s real_data_effects_test_results
rmdir /s rpl_analysis_results
rmdir /s simple_effects_test_results
rmdir /s utility_function_analysis_results
rmdir /s utility_function_results

# 기타 디렉토리 삭제
rmdir /s sem_individual_implementation
rmdir /s test_diagram

# processed_data 하위 디렉토리 정리
rmdir /s processed_data\dce_data
rmdir /s processed_data\docs
rmdir /s processed_data\examples
rmdir /s processed_data\modules
rmdir /s processed_data\survey_data
rmdir /s processed_data\survey_data_backup_restore
rmdir /s processed_data\tests

# 캐시 파일 삭제
rmdir /s __pycache__
rmdir /s scripts\__pycache__
rmdir /s src\__pycache__
```

### 3단계: final_english_charts 처리
이 디렉토리는 중요한 최종 결과를 포함하므로 신중하게 처리:

**옵션 A: 보존 (권장)**
```bash
# 현재 위치에 그대로 두기 (참조용)
```

**옵션 B: 이동**
```bash
# results/current/final_charts/로 이동
mkdir results\current\final_charts
xcopy final_english_charts\* results\current\final_charts\ /s
rmdir /s final_english_charts
```

## 📊 정리 후 예상 구조

정리 완료 후 깔끔한 구조:
```
Sugar_substitue_2025_Heo/
├── main.py                 # 통합 실행 시스템
├── config.py               # 설정 관리
├── README.md               # 프로젝트 문서
├── Raw data/               # 원본 데이터
├── data/                   # 체계적인 데이터 관리
│   ├── raw/
│   ├── processed/
│   └── config/
├── src/                    # 모듈화된 소스 코드
│   ├── analysis/
│   ├── visualization/
│   └── utils/
├── scripts/                # 통합 실행 스크립트
│   ├── run_factor_analysis.py
│   ├── run_reliability_analysis.py
│   ├── run_path_analysis.py
│   └── manage_results.py
├── results/                # 결과 관리 시스템
│   ├── current/
│   └── archive/
├── docs/                   # 문서화
├── logs/                   # 로그 관리
├── tests/                  # 테스트 코드
├── notebooks/              # 분석 노트북
├── final_english_charts/   # 최종 결과 (선택적 보존)
└── processed_data/         # 설정 파일만 유지
    └── reverse_items_config.json
```

## 🎯 정리 후 확인사항

### 1. 기능 테스트
```bash
python main.py --help
python main.py --results
```

### 2. 데이터 접근 확인
```bash
python -c "import pandas as pd; print(pd.read_csv('data/processed/survey/health_concern.csv').shape)"
```

### 3. 결과 디렉토리 확인
```bash
ls -la results/current/
ls -la results/archive/
```

## ⚠️ 주의사항

1. **Raw data/ 디렉토리는 절대 삭제하지 마세요** - 원본 데이터입니다
2. **results/archive/ 확인** - 중요한 과거 결과가 백업되어 있는지 확인
3. **단계적 정리** - 한 번에 모든 것을 삭제하지 말고 단계적으로 진행
4. **백업 확인** - 정리 전에 중요한 파일이 새로운 위치에 있는지 확인

## 🚀 정리 완료 후

정리가 완료되면 다음 명령으로 새로운 시스템을 사용하세요:

```bash
# 대화형 메뉴
python main.py

# 개별 분석
python main.py --factor
python main.py --reliability
python main.py --path

# 전체 분석
python main.py --all
```

---

**🎉 축하합니다! 깔끔하고 체계적인 코드베이스가 완성됩니다!**
