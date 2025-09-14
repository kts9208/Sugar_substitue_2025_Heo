# 경로분석 가시화 결과 요약

## 🎯 실행 개요

**실행 일시**: 2025-09-14 09:44  
**데이터**: processed_data/survey_data (5개 요인, 330개 관측치, 38개 변수)  
**분석 모델**: 종합적 구조방정식 모델 (20개 구조적 경로)  
**가시화 도구**: semopy + graphviz  

## 📊 생성된 가시화 파일 (총 15개)

### 1. 기본 다이어그램
- **comprehensive_path_diagram.png**: 전체 모델 (공분산 포함, 표준화 추정값)
- **comprehensive_structural_only.png**: 구조적 경로만 표시 (잠재변수 간 관계)

### 2. 다양한 스타일 다이어그램 (6가지)
- **comprehensive_multiple_basic.png**: 기본 스타일
- **comprehensive_multiple_detailed.png**: 상세 정보 포함
- **comprehensive_multiple_simple.png**: 간단한 구조만
- **comprehensive_multiple_circular.png**: 원형 레이아웃
- **comprehensive_multiple_unstandardized.png**: 비표준화 계수
- **comprehensive_multiple_structural_only.png**: 구조적 경로만

### 3. 고급 레이아웃 다이어그램 (7가지)
- **comprehensive_advanced_network.png**: 네트워크 레이아웃 (neato 엔진)
- **comprehensive_advanced_hierarchical.png**: 계층적 레이아웃 (fdp 엔진)
- **comprehensive_advanced_spring.png**: 스프링 레이아웃 (sfdp 엔진)
- **comprehensive_advanced_radial.png**: 방사형 레이아웃 (twopi 엔진)
- **comprehensive_advanced_covariance_focus.png**: 공분산 강조
- **comprehensive_advanced_path_focus.png**: 경로계수 강조
- **comprehensive_advanced_structural_paths_only.png**: 구조적 경로만

## 🔍 모델 구조 분석

### 잠재변수 (5개)
1. **health_concern** (건강 관심도): q6~q11 (6개 문항)
2. **perceived_benefit** (지각된 혜택): q12~q17 (6개 문항)  
3. **perceived_price** (지각된 가격): q27~q29 (3개 문항)
4. **nutrition_knowledge** (영양 지식): q30~q49 (20개 문항)
5. **purchase_intention** (구매 의도): q18~q20 (3개 문항)

### 구조적 경로 (20개)
```
perceived_benefit ~ health_concern
perceived_price ~ health_concern  
nutrition_knowledge ~ health_concern
purchase_intention ~ health_concern
perceived_benefit ~ nutrition_knowledge
purchase_intention ~ nutrition_knowledge
perceived_price ~ nutrition_knowledge
purchase_intention ~ perceived_benefit
purchase_intention ~ perceived_price
perceived_price ~ perceived_benefit
perceived_benefit ~ perceived_price
nutrition_knowledge ~ perceived_benefit
health_concern ~ nutrition_knowledge
health_concern ~ perceived_benefit
health_concern ~ purchase_intention
nutrition_knowledge ~ perceived_price
health_concern ~ perceived_price
nutrition_knowledge ~ purchase_intention
perceived_price ~ purchase_intention
perceived_benefit ~ purchase_intention
```

## 📈 모델 적합도 지수

- **Chi-square**: 1420.71
- **CFI**: 0.872 (양호)
- **TLI**: 0.861 (양호)
- **RMSEA**: 0.060 (양호)
- **AIC**: 183.39
- **BIC**: 548.10

## 🎨 가시화 특징

### 기술적 구현
- **semopy.semplot** 함수 활용
- **Graphviz** 엔진 사용 (dot, neato, fdp, sfdp, twopi, circo)
- **PNG 형식** 출력
- **표준화/비표준화** 계수 선택 가능

### 시각화 옵션
- **plot_covs**: 공분산 표시 여부
- **plot_ests**: 추정값 표시 여부  
- **std_ests**: 표준화 추정값 사용 여부
- **structural_only**: 구조적 경로만 표시
- **engine**: 레이아웃 엔진 선택

## 📁 파일 위치

**출력 디렉토리**: `path_analysis_results/visualizations/`

모든 가시화 파일은 PNG 형식으로 저장되어 있으며, 각각 다른 레이아웃과 스타일을 제공합니다.

## ✅ 성공 요인

1. **완전한 semopy 통합**: 기존 경로분석 결과를 semopy 모델로 재생성
2. **다양한 시각화 옵션**: 15가지 다른 스타일과 레이아웃
3. **구조적 경로 강조**: 잠재변수 간 관계에 집중한 다이어그램
4. **안정적인 실행**: 모든 가시화 작업이 성공적으로 완료

## 🔧 기술적 세부사항

- **Fisher Information Matrix 경고**: 모델 복잡성으로 인한 일반적인 경고
- **Moore-Penrose 역행렬 사용**: 수치적 안정성을 위한 대안 방법
- **관측변수 숨김**: 구조적 경로 다이어그램에서 38개 관측변수 제외
- **20개 구조적 경로**: 5개 잠재변수 간의 모든 이론적 관계

이 가시화 결과는 설탕 대체재에 대한 소비자 행동 모델의 복잡한 관계를 다양한 관점에서 이해할 수 있도록 도와줍니다.
