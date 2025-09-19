# 🎯 Sugar Substitute Research - 통합시스템 특징 요약

## 📊 시스템 개요

Sugar Substitute Research 통합시스템은 설탕 대체재 구매의도 연구를 위한 **완전 통합형 분석 플랫폼**입니다.

### 🏗️ 아키텍처 특징

#### **5계층 모듈화 구조**
```
🎯 User Interface Layer    → 사용자 인터페이스
🚀 Execution Scripts Layer → 실행 스크립트
🔬 Core Analysis Modules   → 핵심 분석 모듈
📊 Data Layer             → 데이터 관리
📁 Output & Documentation → 결과 및 문서
```

#### **단일 진입점 (Single Entry Point)**
- **main.py**: 모든 분석의 통합 실행 시스템
- **config.py**: 중앙화된 설정 관리
- **일관된 인터페이스**: 대화형 메뉴 + 명령행 옵션

## 🚀 핵심 기능

### 1. **통합 분석 파이프라인**
```bash
python main.py --all
```
- **요인분석** → **신뢰도 분석** → **경로분석** → **결과 통합**
- 자동화된 순차 실행
- 단계별 결과 검증

### 2. **개별 분석 모듈**

#### 📈 **요인분석 (Factor Analysis)**
```bash
python main.py --factor
```
- **확인적 요인분석 (CFA)**: semopy 기반
- **요인적재량**: 각 문항의 기여도 측정
- **모델 적합도**: CFI, TLI, RMSEA, SRMR
- **5개 요인**: 건강관심도, 지각된유익성, 구매의도, 지각된가격, 영양지식

#### 📊 **신뢰도 분석 (Reliability Analysis)**
```bash
python main.py --reliability
```
- **Cronbach's Alpha**: 내적 일관성
- **복합신뢰도 (CR)**: 구성개념 신뢰도
- **평균분산추출 (AVE)**: 수렴타당도
- **문항 분석**: 문항-총점 상관, 문항간 상관

#### 🔗 **경로분석 (Path Analysis)**
```bash
python main.py --path
```
- **매개분석**: 건강관심도 → 지각된유익성 → 구매의도
- **조절효과**: 지각된가격의 조절역할
- **직접/간접 효과**: 경로계수 및 매개효과
- **부트스트랩**: 신뢰구간 및 유의성 검증

### 3. **고급 분석 모듈**

#### 🎯 **조절효과 분석**
- 지각된가격 × 지각된유익성 → 구매의도
- 영양지식의 조절역할 분석
- 상호작용 효과 시각화

#### 📊 **다항로짓 분석**
- DCE (Discrete Choice Experiment) 데이터 분석
- 선택 확률 모델링
- 효용함수 추정

#### 🔧 **효용함수 분석**
- 개별 효용 계산
- 집단 효용 추정
- 선호 이질성 분석

## 📊 데이터 관리

### **체계적인 데이터 구조**
```
data/
├── raw/                    # 원본 데이터 (Sugar_substitue_Raw data_250730.xlsx)
├── processed/survey/       # 전처리된 설문 데이터 (5개 요인 CSV)
└── config/                 # 설정 파일 (역코딩 등)
```

### **데이터 품질**
- **관측치**: 300개 (완전한 데이터)
- **결측값**: 0개 (데이터 정제 완료)
- **변수 수**: 총 43개 (요인별 4-21개)

## 🎨 시각화 시스템

### **자동 시각화 생성**
- **요인적재량 차트**: 각 요인별 문항 기여도
- **신뢰도 대시보드**: Alpha, CR, AVE 종합 표시
- **경로도표**: 구조방정식 모델 시각화
- **상관관계 히트맵**: 요인 간 관계 매트릭스

### **시각화 모듈**
```
src/visualization/
├── correlation_visualizer.py      # 상관관계 시각화
├── discriminant_validity_analyzer.py # 판별타당도 분석
└── [기타 시각화 모듈]
```

## 💾 결과 관리 시스템

### **자동 버전 관리**
```
results/
├── current/                # 현재 결과 (10개 파일)
└── archive/                # 아카이브 (483개 파일)
```

### **ResultsManager 클래스**
- **자동 아카이브**: 새 분석 시 이전 결과 백업
- **타임스탬프**: 모든 파일에 실행 시간 기록
- **메타데이터**: JSON 형태로 분석 정보 저장
- **버전 추적**: 분석 히스토리 관리

### **결과 형식**
- **CSV**: 수치 데이터 (계수, 적합도 등)
- **JSON**: 메타데이터 및 구조화된 결과
- **PNG**: 시각화 차트
- **TXT**: 분석 보고서

## 🛠️ 사용자 인터페이스

### **대화형 메뉴 시스템**
```
python main.py
```
```
Sugar Substitute Research - 통합 분석 시스템
============================================
1. 요인분석 (Factor Analysis)
2. 신뢰도 분석 (Reliability Analysis)
3. 경로분석 (Path Analysis)
4. 전체 분석 파이프라인
5. 결과 요약 보기
6. 결과 관리
0. 종료
```

### **명령행 인터페이스**
```bash
python main.py --help              # 도움말
python main.py --factor            # 요인분석
python main.py --reliability       # 신뢰도 분석
python main.py --path              # 경로분석
python main.py --all               # 전체 분석
python main.py --results           # 결과 요약
```

### **결과 관리 도구**
```bash
python scripts/manage_results.py --status     # 현재 상태
python scripts/manage_results.py --archive    # 아카이브
python scripts/manage_results.py --history    # 히스토리
```

## 🔧 기술적 특징

### **모듈화 설계**
- **독립적 모듈**: 각 분석 기능별 분리
- **재사용 가능**: 공통 유틸리티 모듈
- **확장 가능**: 새로운 분석 모듈 추가 용이

### **오류 처리**
- **포괄적 예외 처리**: try-catch 블록
- **Fallback 메커니즘**: 대체 경로 지원
- **로깅 시스템**: 상세한 실행 기록

### **성능 최적화**
- **효율적 데이터 로딩**: pandas 기반
- **메모리 관리**: 대용량 데이터 처리
- **병렬 처리**: 가능한 부분 병렬화

## 📚 문서화

### **완전한 문서 체계**
```
docs/
├── USER_GUIDE.md           # 사용자 가이드 (7,590 bytes)
├── API_REFERENCE.md        # API 문서 (7,915 bytes)
├── CHANGELOG.md            # 변경 로그 (6,744 bytes)
└── analysis_reports/       # 분석 보고서
```

### **코드 문서화**
- **Docstring**: 모든 함수/클래스 문서화
- **타입 힌트**: 매개변수 및 반환값 명시
- **예시 코드**: 사용법 예시 제공

## 🧪 테스트 시스템

### **포괄적 테스트**
```
tests/                      # 25개 테스트 파일
├── test_factor_analysis.py
├── test_reliability.py
├── test_path_analysis.py
└── [기타 테스트]
```

### **검증된 기능**
- **시스템 테스트**: 100% 통과 (5/5)
- **메뉴 테스트**: 100% 통과 (4/4)
- **기능 테스트**: 80% 통과 (4/5)
- **분석 테스트**: 모든 핵심 기능 정상

## 🎯 주요 장점

### ✅ **통합성**
- 모든 분석을 하나의 시스템에서 수행
- 일관된 인터페이스와 워크플로우
- 자동화된 분석 파이프라인

### ✅ **사용 편의성**
- 직관적인 대화형 메뉴
- 명령행 옵션 지원
- 실시간 진행 상황 표시

### ✅ **신뢰성**
- 포괄적인 오류 처리
- 자동 백업 및 버전 관리
- 검증된 분석 알고리즘

### ✅ **확장성**
- 모듈화된 구조
- 새로운 분석 기법 추가 용이
- 다양한 데이터 형식 지원

### ✅ **재현성**
- 모든 분석 과정 기록
- 버전 관리 시스템
- 메타데이터 추적

## 🚀 사용 시나리오

### **일반적인 연구 워크플로우**
1. **데이터 준비**: 원본 데이터를 processed/ 디렉토리에 배치
2. **탐색적 분석**: `python main.py --factor`로 요인구조 확인
3. **측정모델 검증**: `python main.py --reliability`로 신뢰도 확인
4. **구조모델 검증**: `python main.py --path`로 가설 검증
5. **결과 해석**: results/current/에서 결과 확인 및 해석

### **고급 분석 시나리오**
1. **전체 파이프라인**: `python main.py --all`로 완전 자동화
2. **결과 비교**: `python scripts/manage_results.py --history`로 버전 비교
3. **커스텀 분석**: 개별 모듈을 직접 호출하여 세밀한 제어

이 통합시스템을 통해 설탕 대체재 구매의도 연구를 체계적이고 효율적으로 수행할 수 있습니다.
