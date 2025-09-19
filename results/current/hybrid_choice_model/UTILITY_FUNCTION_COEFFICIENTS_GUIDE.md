# 📊 효용함수 계수 확인 가이드

## 🎯 효용함수 계수 위치 안내

하이브리드 선택모델의 **효용함수 계수(베타 계수)**는 다음 결과 파일들에서 확인할 수 있습니다:

---

## 📁 **1. JSON 파일 (가장 상세한 정보)**

### **파일 위치**
```
results/current/hybrid_choice_model/hybrid_analysis_[모델명]_[타임스탬프].json
```

### **최신 파일 예시**
```
hybrid_analysis_multinomial_logit_20250919_115014.json
```

### **효용함수 계수 위치**
JSON 파일 내에서 다음 경로에 위치:
```json
{
  "model_fit": {
    "utility_function": {
      "coefficients": {
        "price": {
          "coefficient": -3.9e-05,
          "standardized_coef": -0.060331,
          "t_stat": "N/A",
          "p_value": "N/A"
        },
        "sugar_content": {
          "coefficient": 0.001848,
          "standardized_coef": 0.068603,
          "t_stat": "N/A", 
          "p_value": "N/A"
        },
        ...
      }
    }
  }
}
```

---

## 📄 **2. TXT 요약 파일 (읽기 쉬운 형태)**

### **파일 위치**
```
results/current/hybrid_choice_model/hybrid_summary_[모델명]_[타임스탬프].txt
```

### **최신 파일 예시**
```
hybrid_summary_multinomial_logit_20250919_115014.txt
```

### **효용함수 계수 섹션**
```
효용함수 계수:
- price: -3.9e-05
- sugar_content: 0.001848
- health_label: -0.006431
- health_concern: -0.00147
- perceived_benefit: 0.003061
- purchase_intention: -0.000739
- perceived_price: -0.000672
- intercept: -0.674243
```

---

## 📊 **3. 효용함수 계수 해석**

### **DCE 속성 계수**

#### **가격 (price): -3.9e-05**
- **음수**: 가격이 높을수록 효용 감소 (경제학적으로 타당)
- **크기**: 가격 1원 증가 시 효용 0.000039 감소
- **해석**: 가격 민감도가 상대적으로 낮음

#### **설탕 함량 (sugar_content): 0.001848**
- **양수**: 설탕 함량이 높을수록 효용 증가
- **해석**: 소비자가 단맛을 선호하는 경향

#### **건강 라벨 (health_label): -0.006431**
- **음수**: 건강 라벨이 있을 때 효용 감소
- **해석**: 예상과 다른 결과 (추가 분석 필요)

### **SEM 요인 계수**

#### **건강관심도 (health_concern): -0.00147**
- **음수**: 건강관심도가 높을수록 효용 감소
- **해석**: 건강을 중시하는 소비자일수록 선택 확률 낮음

#### **지각된유익성 (perceived_benefit): 0.003061**
- **양수**: 유익성을 높게 인식할수록 효용 증가
- **해석**: 제품의 유익성 인식이 선택에 긍정적 영향

#### **구매의도 (purchase_intention): -0.000739**
- **음수**: 구매의도가 높을수록 효용 감소
- **해석**: 역설적 결과 (모델 재검토 필요)

#### **지각된가격 (perceived_price): -0.000672**
- **음수**: 가격을 높게 인식할수록 효용 감소
- **해석**: 가격 인식이 선택에 부정적 영향

#### **절편 (intercept): -0.674243**
- **음수**: 기본 효용이 음수
- **해석**: 다른 조건이 같을 때 선택하지 않을 확률이 높음

---

## 🔍 **4. 계수 유형별 설명**

### **원래 계수 (coefficient)**
- 실제 변수 단위에서의 계수
- 변수 1단위 변화 시 효용 변화량
- **실무적 해석에 사용**

### **표준화 계수 (standardized_coef)**
- 표준화된 변수에서의 계수
- 변수 간 상대적 중요도 비교 가능
- **변수 중요도 비교에 사용**

---

## 📈 **5. 지불의사액 (WTP) 계산**

### **계산 공식**
```
WTP = -β_속성 / β_가격
```

### **예시 계산**
```
설탕함량 1% 증가에 대한 WTP:
WTP = -0.001848 / (-0.000039) = 47.38원

건강라벨에 대한 WTP:
WTP = -(-0.006431) / (-0.000039) = -164.90원
```

---

## 🎯 **6. 권장 확인 순서**

### **1단계: TXT 파일로 빠른 확인**
```bash
# 최신 요약 파일 확인
cat results/current/hybrid_choice_model/hybrid_summary_multinomial_logit_20250919_115014.txt
```

### **2단계: JSON 파일로 상세 확인**
```python
import json

# JSON 파일 로드
with open('results/current/hybrid_choice_model/hybrid_analysis_multinomial_logit_20250919_115014.json', 'r') as f:
    results = json.load(f)

# 효용함수 계수 추출
coefficients = results['model_fit']['utility_function']['coefficients']
for var, coef_info in coefficients.items():
    print(f"{var}: {coef_info['coefficient']}")
```

### **3단계: 계수 해석 및 WTP 계산**
```python
# 가격 계수
price_coef = coefficients['price']['coefficient']

# 각 속성별 WTP 계산
for var, coef_info in coefficients.items():
    if var != 'price' and var != 'intercept':
        wtp = -coef_info['coefficient'] / price_coef
        print(f"{var} WTP: {wtp:.2f}원")
```

---

## ⚠️ **7. 주의사항**

### **통계적 유의성**
- 현재 t-통계량과 p-값이 "N/A"로 표시됨
- 실제 연구에서는 통계적 유의성 검정 필요

### **모델 적합도**
- Rho-squared: 0.001 (매우 낮음)
- 모델 개선 또는 추가 변수 고려 필요

### **계수 부호**
- 일부 계수의 부호가 이론적 예상과 다름
- 데이터 품질 및 모델 설정 재검토 필요

---

## 📞 **8. 추가 분석 방법**

### **모델 비교**
```bash
# 다른 모델과 계수 비교
python scripts/run_hybrid_choice_analysis.py --model random_parameters_logit
```

### **탄력성 분석**
```python
# 속성별 탄력성 계산
elasticity = coefficient * mean_value * (1 - choice_probability)
```

### **시나리오 분석**
```python
# 속성 변화 시 선택 확률 변화 예측
new_utility = sum(coef * new_value for coef, new_value in zip(coefficients, new_attributes))
new_probability = exp(new_utility) / (1 + exp(new_utility))
```

---

## 🎉 **결론**

**효용함수 계수는 다음 파일에서 확인 가능합니다:**

1. **📄 빠른 확인**: `hybrid_summary_[모델명]_[타임스탬프].txt`
2. **📊 상세 분석**: `hybrid_analysis_[모델명]_[타임스탬프].json`
3. **📈 추가 계산**: Python 스크립트로 WTP, 탄력성 등 계산

**가장 최신 파일을 확인하여 정확한 효용함수 계수를 얻으시기 바랍니다!**
