# 판별타당도 검증 모듈 (Discriminant Validity Analysis Module)

## 📋 개요
이 모듈은 상관관계 분석 결과와 신뢰도 분석 결과를 불러와서 요인간 상관계수와 AVE의 제곱근을 비교하여 판별타당도를 검증하는 독립적인 분석 도구입니다.

## 🎯 주요 기능
1. **자동 데이터 로드**: 기존 분석 결과 파일들을 자동으로 찾아서 로드
2. **Fornell-Larcker 기준 검증**: 각 요인의 AVE 제곱근과 상관계수 비교
3. **종합적 시각화**: 다양한 차트와 그래프로 결과 표현
4. **상세 보고서 생성**: 해석과 권고사항이 포함된 분석 보고서
5. **CSV 결과 출력**: 추가 분석을 위한 구조화된 데이터

## 📁 모듈 구성

### 핵심 파일
- **`discriminant_validity_analyzer.py`**: 메인 분석 클래스
- **`run_discriminant_validity_analysis.py`**: 실행 스크립트
- **`test_discriminant_validity.py`**: 테스트 스크립트

### 결과 디렉토리
- **`discriminant_validity_results/`**: 모든 분석 결과 저장

## 🚀 사용법

### 1. 기본 실행
```bash
python run_discriminant_validity_analysis.py
```

### 2. 프로그래밍 방식 사용
```python
from discriminant_validity_analyzer import DiscriminantValidityAnalyzer

# 자동 파일 찾기
analyzer = DiscriminantValidityAnalyzer()

# 또는 특정 파일 지정
analyzer = DiscriminantValidityAnalyzer(
    correlation_file="path/to/correlations.csv",
    reliability_file="path/to/reliability.csv"
)

# 전체 분석 실행
analyzer.run_complete_analysis()
```

### 3. 개별 기능 사용
```python
# 데이터 로드
analyzer.load_data()

# AVE 제곱근 매트릭스 생성
analyzer.create_ave_sqrt_matrix()

# 판별타당도 분석
results = analyzer.analyze_discriminant_validity()

# 시각화 생성
analyzer.visualize_discriminant_validity()

# 보고서 생성
analyzer.generate_report()
```

## 📊 생성되는 결과물

### 시각화 파일
1. **correlation_vs_ave_comparison.png**
   - 상관계수와 AVE 제곱근 비교 히트맵
   
2. **discriminant_validity_matrix.png**
   - 판별타당도 검증 결과 매트릭스
   
3. **discriminant_validity_dashboard.png**
   - 종합 대시보드 (파이차트, 막대그래프, 히스토그램, 요약)
   
4. **discriminant_validity_violations.png** (위반 시에만)
   - 위반 사항 상세 시각화

### 데이터 파일
1. **discriminant_validity_report_[timestamp].txt**
   - 상세 분석 보고서
   
2. **discriminant_validity_results_[timestamp].csv**
   - 검증 결과 데이터
   
3. **correlation_ave_comparison_matrix_[timestamp].csv**
   - 상관계수와 AVE 제곱근 비교 매트릭스

## 🔍 분석 방법론

### Fornell-Larcker 기준
판별타당도 검증의 표준 방법으로, 다음 조건을 만족해야 합니다:
- **조건**: √AVE_i > |r_ij| (모든 i ≠ j에 대해)
- **의미**: 각 요인이 자신의 측정항목들과 더 많은 분산을 공유해야 함

### 검증 과정
1. 각 요인의 AVE 제곱근 계산
2. 모든 요인 쌍의 상관계수 추출
3. 각 쌍에 대해 min(√AVE_i, √AVE_j) > |r_ij| 검증
4. 위반 사항 식별 및 분석

## 📈 현재 분석 결과

### 요약
- **전체 검증 쌍**: 10개
- **유효한 쌍**: 9개 (90.0%)
- **위반 쌍**: 1개 (10.0%)
- **전체 판별타당도**: ❌ 미달성

### 위반 사항
**perceived_benefit vs purchase_intention**
- 상관계수: 0.892
- 최소 AVE 제곱근: 0.780
- 위반 크기: 0.112

### 해석
지각된 혜택(perceived_benefit)과 구매의도(purchase_intention) 간의 상관관계가 매우 높아서 두 구성개념이 충분히 구별되지 않음을 의미합니다.

## 🛠️ 기술적 특징

### 자동화된 파일 찾기
- 최신 상관관계 분석 결과 자동 탐지
- 최신 신뢰도 분석 결과 자동 탐지
- 타임스탬프 기반 파일 선택

### 강건한 오류 처리
- 파일 존재 여부 확인
- 데이터 형식 검증
- 상세한 오류 메시지

### 확장 가능한 구조
- 모듈화된 클래스 설계
- 새로운 판별타당도 기준 추가 가능
- 커스텀 시각화 옵션

## 🔧 요구사항

### Python 패키지
```
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

### 입력 데이터 형식
1. **상관관계 데이터**: CSV 파일, 요인명을 인덱스와 컬럼으로 하는 상관계수 매트릭스
2. **신뢰도 데이터**: CSV 파일, Factor, AVE, Sqrt_AVE 컬럼 포함

## 🧪 테스트

### 테스트 실행
```bash
python test_discriminant_validity.py
```

### 테스트 항목
- 분석기 초기화
- 데이터 로드
- AVE 제곱근 매트릭스 생성
- 판별타당도 분석
- 비교 매트릭스 생성
- 출력 파일 확인

## 📚 참고문헌
- Fornell, C., & Larcker, D. F. (1981). Evaluating structural equation models with unobservable variables and measurement error. *Journal of Marketing Research*, 18(1), 39-50.
- Hair, J. F., Black, W. C., Babin, B. J., & Anderson, R. E. (2019). *Multivariate Data Analysis* (8th ed.). Cengage Learning.

## 🤝 기여 및 개선사항
- HTMT 비율 계산 기능 추가
- 교차타당도 검증 기능
- 다양한 시각화 옵션
- 국제화 지원 (영문 보고서)

---
*최종 업데이트: 2025-09-06*
