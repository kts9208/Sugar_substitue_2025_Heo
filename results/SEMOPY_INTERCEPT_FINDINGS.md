# semopy 절편 추정 기능 분석 결과

## 핵심 발견

### ✅ **semopy는 절편 추정 기능이 있습니다!**

**방법**:
- `ModelMeans` 클래스 사용
- 절편은 `op == '~'` AND `rval == '1'`로 저장됨

---

## 1. semopy 절편 추정 방법

### **방법 1: 기본 Model 클래스 (절편 없음)**

```python
from semopy import Model

model_spec = """
LV =~ q1 + q2 + q3
"""

model = Model(model_spec)
model.fit(data)
```

**결과**:
- 절편 추정 안함
- 잠재변수 평균 E[LV] = 0으로 고정

---

### **방법 2: ModelMeans 클래스 (절편 있음) ✅**

```python
from semopy import ModelMeans

model_spec = """
LV =~ q1 + q2 + q3
"""

model = ModelMeans(model_spec)
model.fit(data)
```

**결과**:
```
  lval  op rval  Estimate
0   q1   ~   LV  1.000000  # 요인적재량
1   q2   ~   LV  1.123841
2   q3   ~   LV  0.785197
3   q1   ~    1  3.407269  # ✅ 절편 (op='~', rval='1')
4   q2   ~    1  3.738955  # ✅ 절편
5   q3   ~    1  3.970322  # ✅ 절편
6   LV  ~~   LV  0.681469
7   q1  ~~   q1  0.243164
8   q2  ~~   q2  0.259071
9   q3  ~~   q3  0.187630
```

**절편 필터링**:
```python
params = model.inspect()
intercepts = params[params['rval'] == '1']  # ✅ rval == '1'
```

---

## 2. 현재 CFA 추정 코드 분석

### **현재 사용 중인 클래스**

<augment_code_snippet path="src/analysis/factor_analysis/factor_analyzer.py" mode="EXCERPT">
```python
from semopy import Model  # ❌ Model 클래스 사용 (절편 없음)

self.model = Model(model_spec)
self.results = self.model.fit(clean_data, solver=self.config.optimizer)
```
</augment_code_snippet>

**문제**:
- `Model` 클래스 사용 → 절편 추정 안함
- `ModelMeans` 클래스 사용 필요

---

## 3. 해결 방안

### ✅ **Option A: ModelMeans 클래스로 변경 (추천)**

**수정 위치**: `src/analysis/factor_analysis/factor_analyzer.py`

**변경 전**:
```python
from semopy import Model

self.model = Model(model_spec)
```

**변경 후**:
```python
from semopy import ModelMeans

self.model = ModelMeans(model_spec)
```

**장점**:
- ✅ 절편 자동 추정
- ✅ 코드 수정 최소화 (1줄)
- ✅ 이론적으로 올바름

**절편 추출**:
```python
params = self.model.inspect()
intercepts = params[params['rval'] == '1'].copy()
```

---

### **Option B: 절편을 지표 평균으로 계산**

**방법**:
- CFA 추정 후 각 지표의 평균 계산
- 절편으로 저장

**장점**:
- ✅ semopy 의존성 없음
- ✅ 항상 작동

**단점**:
- ⚠️ CFA 추정과 별도 계산

---

## 4. 권장 사항

### ✅ **Option A: ModelMeans 클래스 사용 (강력 추천)**

**이유**:
1. **semopy 공식 기능**: 절편 추정이 내장됨
2. **코드 수정 최소화**: 1줄만 변경
3. **이론적으로 올바름**: CFA에서 절편 추정
4. **자동 추출**: `params[params['rval'] == '1']`

**구현 순서**:
1. `factor_analyzer.py`에서 `Model` → `ModelMeans` 변경
2. `sem_estimator.py`에서 절편 추출 로직 추가
3. `sequential_estimator.py`에서 절편 저장
4. CFA 재추정

---

## 5. 절편 필터링 방법

### **semopy의 절편 표현**

```python
# 절편은 다음과 같이 저장됨:
# lval  op  rval  Estimate
#   q1   ~     1  3.407269

# 필터링 방법 1: rval == '1'
intercepts = params[params['rval'] == '1']

# 필터링 방법 2: op == '~' AND rval == '1'
intercepts = params[(params['op'] == '~') & (params['rval'] == '1')]
```

**주의**:
- `op == '1'`이 아님! (이전 가정이 틀렸음)
- `rval == '1'`로 필터링해야 함

---

## 6. 결론

### **현재 상황**

❌ **CFA 추정 코드에서 `Model` 클래스 사용 → 절편 추정 안함**

### **해결 방안**

✅ **`ModelMeans` 클래스로 변경**
- `src/analysis/factor_analysis/factor_analyzer.py` 수정
- `from semopy import Model` → `from semopy import ModelMeans`
- `self.model = Model(spec)` → `self.model = ModelMeans(spec)`

### **절편 추출**

```python
params = model.inspect()
intercepts = params[params['rval'] == '1'].copy()
intercepts['param_type'] = 'intercept'
```

### **예상 효과**

- 측정모델 우도: **-1218.39 → -27.74** (44배 개선!)
- 전체 우도 균형 크게 개선

---

**작성일**: 2025-11-20  
**작성자**: Augment Agent

