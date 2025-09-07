# 판별타당도 분석 결과 (Discriminant Validity Analysis Results)

## 개요
이 디렉토리는 상관관계 분석 결과와 신뢰도 분석 결과를 바탕으로 수행된 판별타당도 검증 결과를 포함합니다.

## 판별타당도란?
판별타당도(Discriminant Validity)는 서로 다른 구성개념들이 실제로 구별되는지를 검증하는 측정모형의 타당도 평가 기준입니다.

### 검증 기준: Fornell-Larcker Criterion
- 각 요인의 AVE 제곱근이 다른 요인들과의 상관계수보다 커야 함
- 이는 각 요인이 자신의 측정항목들과 더 많은 분산을 공유함을 의미
- 공식: √AVE_i > |r_ij| (모든 i ≠ j에 대해)

## 분석 결과 요약

### 전체 결과
- **전체 검증 쌍**: 10개
- **유효한 쌍**: 9개 (90.0%)
- **위반 쌍**: 1개 (10.0%)
- **전체 판별타당도**: ❌ 미달성

### 요인별 AVE 제곱근
| 요인 | AVE 제곱근 |
|------|-----------|
| health_concern | 0.846 |
| perceived_benefit | 0.780 |
| purchase_intention | 1.026 |
| perceived_price | 0.815 |
| nutrition_knowledge | 0.716 |

### 판별타당도 위반 사항
**위반**: perceived_benefit vs purchase_intention
- 상관계수: 0.892
- 최소 AVE 제곱근: 0.780
- 위반 크기: 0.112

이는 지각된 혜택(perceived_benefit)과 구매의도(purchase_intention) 간의 상관관계가 매우 높아서 두 구성개념이 충분히 구별되지 않음을 의미합니다.

## 생성된 파일들

### 📊 시각화 파일
1. **correlation_vs_ave_comparison.png**
   - 상관계수와 AVE 제곱근 비교 히트맵
   - 왼쪽: 요인간 상관계수 (하삼각)
   - 오른쪽: AVE 제곱근 (대각선)

2. **discriminant_validity_matrix.png**
   - 판별타당도 검증 결과 매트릭스
   - 초록색: 유효 (Valid)
   - 빨간색: 위반 (Invalid)

3. **discriminant_validity_dashboard.png**
   - 종합 대시보드
   - 전체 결과 파이차트, AVE 제곱근 막대그래프, 상관계수 분포, 요약 정보

4. **discriminant_validity_violations.png**
   - 위반 사항 상세 시각화
   - 상관계수 vs AVE 제곱근 비교
   - 위반 크기 막대그래프

### 📄 보고서 및 데이터 파일
1. **discriminant_validity_report_[timestamp].txt**
   - 상세 분석 보고서
   - 분석 개요, 검증 결과, 해석 및 권고사항 포함

2. **discriminant_validity_results_[timestamp].csv**
   - 검증 결과 데이터
   - 각 요인 쌍별 상관계수, AVE 제곱근, 검증 결과 포함

3. **correlation_ave_comparison_matrix_[timestamp].csv**
   - 상관계수와 AVE 제곱근 비교 매트릭스
   - 대각선: √AVE 값
   - 비대각선: 상관계수 값

## 해석 및 권고사항

### 현재 상황
- 90%의 요인 쌍이 판별타당도 기준을 만족
- perceived_benefit과 purchase_intention 간의 높은 상관관계(0.892)로 인한 위반

### 권고사항
1. **측정항목 재검토**
   - perceived_benefit과 purchase_intention의 측정항목들을 검토
   - 개념적 중복을 제거하거나 더 명확하게 구분되는 항목으로 수정

2. **요인 구조 재정의**
   - 두 요인을 하나의 상위 요인으로 통합 고려
   - 또는 더 세분화된 하위 요인으로 분리

3. **추가 타당도 검증**
   - HTMT (Heterotrait-Monotrait) 비율 계산
   - 교차타당도 검증 실시

4. **이론적 재검토**
   - 지각된 혜택과 구매의도의 개념적 관계 재정의
   - 선행연구를 통한 이론적 근거 보강

## 사용법

### 분석 재실행
```bash
python run_discriminant_validity_analysis.py
```

### 커스텀 분석
```python
from discriminant_validity_analyzer import DiscriminantValidityAnalyzer

# 특정 파일 지정
analyzer = DiscriminantValidityAnalyzer(
    correlation_file="path/to/correlation.csv",
    reliability_file="path/to/reliability.csv"
)

# 분석 실행
analyzer.run_complete_analysis()
```

## 참고문헌
- Fornell, C., & Larcker, D. F. (1981). Evaluating structural equation models with unobservable variables and measurement error. Journal of Marketing Research, 18(1), 39-50.
- Hair, J. F., Black, W. C., Babin, B. J., & Anderson, R. E. (2019). Multivariate Data Analysis (8th ed.). Cengage Learning.

---
*분석 일시: 2025-09-06 14:51:39*
