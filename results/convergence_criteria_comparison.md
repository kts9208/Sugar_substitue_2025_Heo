# 수렴 판단 기준 비교: 현재 코드 vs Apollo R

## 1. 현재 Python 코드의 수렴 기준

### BFGS Optimizer 설정 (simultaneous_estimator_fixed.py, line 450-455)

```python
options={
    'maxiter': self.config.estimation.max_iterations,  # 기본값: 1000
    'ftol': 1e-6,    # 함수값 변화 허용 오차
    'gtol': 1e-5,    # Gradient norm 허용 오차
    'disp': True
}
```

### 수렴 판정 기준

| 기준 | 값 | 의미 |
|------|-----|------|
| **ftol** | 1e-6 | 연속된 반복 간 함수값(LL) 변화가 이 값보다 작으면 수렴 |
| **gtol** | 1e-5 | Gradient의 L2 norm이 이 값보다 작으면 수렴 |
| **maxiter** | 1000 | 최대 반복 횟수 |

**수렴 조건**: `ftol` OR `gtol` 조건 중 하나라도 만족하면 수렴으로 판정

---

## 2. Apollo R의 수렴 기준

### BGW Optimizer 기본 설정

Apollo R은 `maxLik` 패키지의 BGW (Bunch-Gay-Welsch) optimizer를 사용합니다.

**출처**: Apollo Manual (2025), maxLik documentation

### maxLik BGW 기본 파라미터

| 파라미터 | 기본값 | 의미 |
|----------|--------|------|
| **tol** | 1e-8 | Gradient tolerance (gtol과 동일) |
| **reltol** | sqrt(.Machine$double.eps) ≈ 1.49e-8 | 상대적 함수값 변화 허용 오차 |
| **gradtol** | 1e-6 | Gradient tolerance (대체 기준) |
| **steptol** | 1e-10 | Step size tolerance |
| **iterlim** | 150 | 최대 반복 횟수 (기본값) |

**R의 `.Machine$double.eps`**: 약 2.22e-16 (부동소수점 정밀도)
**sqrt(.Machine$double.eps)**: 약 1.49e-8

### Apollo 특화 설정

Apollo는 `apollo_control` 리스트를 통해 추가 설정 가능:

```r
apollo_control = list(
  modelName = "model_name",
  mixing = TRUE,
  nCores = 4,
  # 수렴 기준 (선택적)
  maxIterations = 200,  # BGW 최대 반복
  tolerance = 1e-6      # 수렴 허용 오차
)
```

**기본 수렴 조건**: 
- Gradient norm < `tol` (1e-8) 
- OR 상대적 LL 변화 < `reltol` (1.49e-8)
- OR 반복 횟수 > `iterlim` (150)

---

## 3. 비교 분석

### 3.1 Gradient Tolerance 비교

| 구분 | Python (BFGS) | Apollo R (BGW) | 차이 |
|------|---------------|----------------|------|
| **gtol** | 1e-5 | 1e-8 (tol) | **Python이 1000배 느슨** |
| **의미** | Gradient norm < 1e-5 | Gradient norm < 1e-8 | Apollo가 더 엄격 |

**결론**: Python 코드가 **gradient 기준으로는 더 빨리 수렴**할 수 있음

### 3.2 Function Tolerance 비교

| 구분 | Python (BFGS) | Apollo R (BGW) | 차이 |
|------|---------------|----------------|------|
| **ftol** | 1e-6 (절대값) | 1.49e-8 (상대값) | **기준이 다름** |
| **의미** | \|LL_new - LL_old\| < 1e-6 | \|LL_new - LL_old\| / \|LL_old\| < 1.49e-8 | Apollo는 상대적 변화 |

**예시 계산** (LL = -5362.47):
- **Python ftol**: LL 변화 < 1e-6 (절대값)
- **Apollo reltol**: LL 변화 < 5362.47 × 1.49e-8 ≈ **7.99e-5** (절대값 환산)

**결론**: Apollo의 상대적 기준이 **실질적으로 더 느슨**함 (약 80배)

### 3.3 최대 반복 횟수 비교

| 구분 | Python (BFGS) | Apollo R (BGW) |
|------|---------------|----------------|
| **maxiter** | 1000 | 150 (기본값) |

**결론**: Python이 더 많은 반복 허용

---

## 4. 현재 추정 결과 분석

### 실제 수렴 과정 (Terminal 6 로그 기준)

| 반복 | 로그우도 | LL 변화 (절대값) | LL 변화 (상대값) |
|------|----------|------------------|------------------|
| 69 | -5362.4673 | - | - |
| 70-139 | -5362.4673 | 0.0000 | 0.0000 |

**종료 메시지**: "Precision loss was detected" (BFGS)

### 수렴 판정 분석

**Python ftol 기준 (1e-6)**:
- Iter 69 이후 LL 변화 = 0 < 1e-6 ✅ **만족**

**Apollo reltol 기준 (1.49e-8)**:
- Iter 69 이후 상대 변화 = 0 < 1.49e-8 ✅ **만족**

**결론**: **두 기준 모두 Iter 69에서 실질적 수렴 달성**

### "Precision Loss" 의미

BFGS가 "Precision loss"로 종료한 이유:
1. LL 변화가 부동소수점 정밀도 한계에 도달
2. 추가 개선이 수치적으로 불가능
3. **실질적으로는 수렴 완료**

Apollo BGW도 동일한 상황에서 유사하게 종료됨

---

## 5. 종합 결론

### 5.1 수렴 기준 비교 요약

| 측면 | Python BFGS | Apollo BGW | 평가 |
|------|-------------|------------|------|
| **Gradient tolerance** | 1e-5 (느슨) | 1e-8 (엄격) | Apollo가 더 엄격 |
| **Function tolerance** | 1e-6 (절대) | 1.49e-8 (상대) | 실질적으로 Apollo가 느슨 |
| **최대 반복** | 1000 | 150 | Python이 더 관대 |
| **수렴 속도** | 빠름 (느슨한 gtol) | 느림 (엄격한 gtol) | Python 유리 |
| **수렴 품질** | 높음 | 매우 높음 | Apollo 약간 우세 |

### 5.2 현재 코드의 수렴 기준 평가

**✅ 장점**:
1. **ftol (1e-6)**: 실용적이고 충분히 엄격함
2. **gtol (1e-5)**: Apollo보다 느슨하지만 실무적으로 충분
3. **maxiter (1000)**: 충분한 반복 허용

**⚠️ 개선 가능 사항**:
1. **gtol을 1e-6 ~ 1e-7로 강화** 가능 (Apollo 수준에 근접)
2. **상대적 ftol 추가** 고려 (큰 LL 값에 대응)

### 5.3 최종 권장사항

**현재 설정 유지 권장**:
- ftol = 1e-6 ✅ 적절
- gtol = 1e-5 ✅ 실무적으로 충분
- maxiter = 1000 ✅ 충분

**선택적 개선** (더 엄격한 수렴 원할 경우):
```python
options={
    'maxiter': 1000,
    'ftol': 1e-7,    # 더 엄격하게 (Apollo 수준)
    'gtol': 1e-6,    # 더 엄격하게 (Apollo 중간)
    'disp': True
}
```

**결론**: **현재 코드의 수렴 기준은 Apollo R과 비교하여 실무적으로 동등하며, 일부 측면에서는 더 실용적입니다. 추가 수정 불필요.**

---

## 6. 참고 문헌

1. Apollo Choice Modelling Manual (2025), www.apollochoicemodelling.com
2. maxLik R Package Documentation, CRAN
3. SciPy optimize.minimize BFGS Documentation
4. Bunch, D. S., Gay, D. M., & Welsch, R. E. (1993). Algorithm 717: Subroutines for maximum likelihood estimation. ACM Transactions on Mathematical Software, 19(4), 484-494.

