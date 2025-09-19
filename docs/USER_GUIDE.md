# Sugar Substitute Research - 사용자 가이드

## 📚 목차

1. [시작하기](#시작하기)
2. [기본 사용법](#기본-사용법)
3. [고급 사용법](#고급-사용법)
4. [결과 해석](#결과-해석)
5. [문제 해결](#문제-해결)

## 🚀 시작하기

### 1단계: 환경 준비

#### 필수 패키지 설치
```bash
pip install pandas numpy scipy semopy matplotlib seaborn
```

#### 프로젝트 디렉토리 설정
```bash
python config.py
```

### 2단계: 데이터 준비

설문조사 데이터를 다음 위치에 배치하세요:
```
data/processed/survey/
├── health_concern.csv          # 건강관심도 데이터
├── perceived_benefit.csv       # 지각된 유익성 데이터
├── purchase_intention.csv      # 구매의도 데이터
├── perceived_price.csv         # 지각된 가격 데이터
└── nutrition_knowledge.csv     # 영양지식 데이터
```

**데이터 형식 요구사항:**
- CSV 형식, UTF-8 인코딩
- 첫 번째 행은 헤더
- 결측값은 빈 셀 또는 'NaN'으로 표시
- 리커트 척도 데이터 (1-7점)

### 3단계: 첫 번째 분석 실행

```bash
python main.py
```

대화형 메뉴가 나타나면 `1`을 선택하여 요인분석부터 시작하세요.

## 📖 기본 사용법

### 대화형 메뉴 사용

```bash
python main.py
```

메뉴 옵션:
- `1`: 요인분석 (Factor Analysis)
- `2`: 신뢰도 분석 (Reliability Analysis)
- `3`: 경로분석 (Path Analysis)
- `4`: 전체 분석 파이프라인
- `5`: 결과 관리
- `6`: 결과 요약 보기
- `0`: 종료

### 명령행 옵션 사용

```bash
# 전체 분석 실행
python main.py --all

# 개별 분석 실행
python main.py --factor          # 요인분석
python main.py --reliability     # 신뢰도 분석
python main.py --path           # 경로분석

# 결과 확인
python main.py --results
```

### 개별 스크립트 실행

더 세밀한 제어가 필요한 경우:

```bash
# 요인분석 - 모든 요인
python scripts/run_factor_analysis.py --all

# 요인분석 - 특정 요인만
python scripts/run_factor_analysis.py --factor health_concern

# 경로분석 - 특정 모델
python scripts/run_path_analysis.py --model simple
python scripts/run_path_analysis.py --model comprehensive
```

## 🎯 고급 사용법

### 1. 결과 버전 관리

#### 현재 상태 확인
```bash
python scripts/manage_results.py --status
```

#### 결과 아카이브
```bash
# 자동 아카이브 (설명 없음)
python scripts/manage_results.py --archive factor_analysis

# 설명과 함께 아카이브
python scripts/manage_results.py --archive path_analysis --description "새로운 모델 테스트"
```

#### 버전 히스토리 확인
```bash
python scripts/manage_results.py --history factor_analysis
```

#### 이전 버전 복원
```bash
python scripts/manage_results.py --restore factor_analysis 20250918_143022
```

### 2. 설정 커스터마이징

`config.py` 파일에서 다음 설정들을 조정할 수 있습니다:

#### 분석 임계값 설정
```python
ANALYSIS_CONFIG = {
    "factor_analysis": {
        "min_loading_threshold": 0.5,      # 최소 요인적재량
        "good_loading_threshold": 0.7,     # 양호한 요인적재량
        "fit_indices_thresholds": {
            "CFI": 0.9,                    # CFI 임계값
            "RMSEA": 0.08                  # RMSEA 임계값
        }
    }
}
```

#### 시각화 설정
```python
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),               # 그림 크기
    "dpi": 300,                           # 해상도
    "korean_font": "Malgun Gothic"        # 한글 폰트
}
```

### 3. 배치 처리

여러 분석을 순차적으로 실행:

```bash
# 핵심 분석만 (요인분석 + 신뢰도 + 경로분석)
python scripts/run_complete_analysis.py --core-only

# 모든 분석 포함
python scripts/run_complete_analysis.py --include-moderation --include-mnl
```

## 📊 결과 해석

### 1. 요인분석 결과

#### Factor Loadings 해석
- **≥ 0.7**: 우수한 적재량 ✅
- **0.5-0.7**: 양호한 적재량 ⚠️
- **< 0.5**: 부족한 적재량 ❌

#### 적합도 지수 해석
- **CFI ≥ 0.9**: 좋은 적합도
- **TLI ≥ 0.9**: 좋은 적합도
- **RMSEA ≤ 0.08**: 수용 가능한 적합도
- **SRMR ≤ 0.08**: 수용 가능한 적합도

### 2. 신뢰도 분석 결과

#### 신뢰도 지표 해석
- **Cronbach's α ≥ 0.7**: 신뢰할 만한 수준
- **CR ≥ 0.7**: 복합신뢰도 양호
- **AVE ≥ 0.5**: 평균분산추출 양호

### 3. 경로분석 결과

#### 경로계수 해석
- **β > 0**: 정(+)의 영향
- **β < 0**: 부(-)의 영향
- **p < 0.05**: 통계적으로 유의한 영향

#### 매개효과 해석
- **직접효과**: X → Y 직접 경로
- **간접효과**: X → M → Y 매개 경로
- **총효과**: 직접효과 + 간접효과

### 4. 판별타당도 결과

#### Fornell-Larcker 기준
- **√AVE > 상관계수**: 판별타당도 확보
- **√AVE ≤ 상관계수**: 판별타당도 문제

## 🔧 문제 해결

### 자주 발생하는 오류

#### 1. 데이터 로딩 오류
```
❌ 설문조사 데이터 디렉토리를 찾을 수 없습니다.
```

**해결책:**
1. 데이터 파일 위치 확인
2. 파일명 확인 (정확한 요인명 사용)
3. 파일 형식 확인 (CSV, UTF-8)

#### 2. 분석 실행 오류
```
❌ 요인분석 실패
```

**해결책:**
1. 로그 파일 확인: `logs/factor_analysis.log`
2. 데이터 품질 확인 (결측값, 이상값)
3. 샘플 크기 확인 (최소 100개 이상 권장)

#### 3. 메모리 부족 오류
```
MemoryError: Unable to allocate array
```

**해결책:**
1. 불필요한 프로그램 종료
2. 데이터 크기 축소
3. 청크 단위 처리 활용

#### 4. 한글 폰트 오류
```
UserWarning: Glyph missing from current font
```

**해결책:**
1. 한글 폰트 설치 확인
2. `config.py`에서 폰트 설정 변경
3. 영문 출력으로 대체

### 성능 최적화 팁

#### 1. 실행 시간 단축
- 부트스트래핑 샘플 수 줄이기 (1000 → 500)
- 불필요한 시각화 비활성화
- 캐시된 결과 재사용

#### 2. 메모리 사용량 최적화
- 분석별 독립 실행
- 중간 결과 정리
- 대용량 데이터 청크 처리

#### 3. 결과 파일 관리
- 정기적인 아카이브 정리
- 불필요한 버전 삭제
- 압축 저장 활용

### 지원 요청

문제가 지속되는 경우:

1. **로그 파일 확인**: `logs/` 디렉토리의 관련 로그
2. **환경 정보 수집**: Python 버전, 패키지 버전
3. **재현 단계 기록**: 오류 발생까지의 정확한 단계
4. **데이터 정보**: 샘플 크기, 변수 수, 데이터 형식

## 📈 모범 사례

### 1. 분석 순서
1. **데이터 탐색** → 2. **요인분석** → 3. **신뢰도 분석** → 4. **경로분석**

### 2. 결과 검증
- 여러 모델 비교
- 적합도 지수 종합 판단
- 이론적 타당성 확인

### 3. 보고서 작성
- 분석 과정 문서화
- 결과 해석 근거 제시
- 한계점 및 제언 포함

## 🎓 학습 자료

### 추천 학습 순서
1. **기본 개념 이해**: 요인분석, 신뢰도, 타당도
2. **실습**: 샘플 데이터로 분석 연습
3. **결과 해석**: 통계 지표의 의미 파악
4. **고급 기능**: 조절효과, 매개효과 분석

### 참고 자료
- **구조방정식 모델링**: Hair et al. (2019)
- **요인분석**: Fabrigar & Wegener (2012)
- **신뢰도 분석**: Cronbach (1951)

---

**Version**: 2.0
**Last Updated**: 2025-09-18
**도움이 필요하시면 언제든지 문의하세요!**
