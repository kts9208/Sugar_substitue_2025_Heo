"""
simultaneous_estimator.py 파일을 수정하여 수정 버전 생성
"""

import re

# 파일 읽기
with open('src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. fit 메서드의 최적화 부분 수정
old_optimize = '''        result = minimize(
            objective,
            initial_params,
            method='BFGS',
            options={
                'maxiter': self.config.estimation.max_iterations,
                'disp': True
            }
        )'''

new_optimize = '''        # Get parameter bounds
        bounds = self._get_parameter_bounds(
            measurement_model, structural_model, choice_model
        )
        
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',  # Changed from BFGS
            bounds=bounds,       # Added bounds
            options={
                'maxiter': self.config.estimation.max_iterations,
                'ftol': 1e-6,    # Added convergence criteria
                'gtol': 1e-5,
                'disp': True
            }
        )'''

content = content.replace(old_optimize, new_optimize)

# 2. _joint_log_likelihood 메서드의 선택모델 부분 수정
old_choice = '''                # 선택모델 로그우도
                ll_choice = choice_model.log_likelihood(
                    person_data,  # 선택 데이터는 개인당 여러 행
                    lv_draw,
                    choice_params
                )
                
                # 결합 로그우도 (로그 공간에서 합산)
                draw_ll = ll_measurement + ll_structural + ll_choice
                draw_lls.append(draw_ll)'''

new_choice = '''                # Panel Product: 개인의 여러 선택 상황에 대한 확률을 곱함
                choice_set_lls = []
                for idx in range(len(person_data)):
                    ll_choice_t = choice_model.log_likelihood(
                        person_data.iloc[idx:idx+1],  # 각 선택 상황
                        lv_draw,
                        choice_params
                    )
                    choice_set_lls.append(ll_choice_t)
                
                # Panel product: log(P1 * P2 * ... * PT) = log(P1) + log(P2) + ... + log(PT)
                ll_choice = sum(choice_set_lls)
                
                # 결합 로그우도 (로그 공간에서 합산)
                draw_ll = ll_measurement + ll_structural + ll_choice
                
                # 수치 안정성: 유한한 값만 사용
                if np.isfinite(draw_ll):
                    draw_lls.append(draw_ll)'''

content = content.replace(old_choice, new_choice)

# 3. logsumexp 부분 수정
old_logsumexp = '''            # 개인별 로그우도: log[(1/R)Σᵣ exp(ll_r)]
            # logsumexp를 사용하여 수치 안정성 확보
            person_ll = logsumexp(draw_lls) - np.log(n_draws)
            total_ll += person_ll'''

new_logsumexp = '''            # 유효한 draws가 없으면 매우 작은 값 반환
            if len(draw_lls) == 0:
                person_ll = -1e10
            else:
                # 개인별 로그우도: log[(1/R)Σᵣ exp(ll_r)]
                # logsumexp를 사용하여 수치 안정성 확보
                person_ll = logsumexp(draw_lls) - np.log(len(draw_lls))
            
            total_ll += person_ll'''

content = content.replace(old_logsumexp, new_logsumexp)

# 4. _get_parameter_bounds 메서드 추가
bounds_method = '''
    
    def _get_parameter_bounds(self, measurement_model,
                              structural_model, choice_model) -> list:
        """
        Parameter bounds for L-BFGS-B
        
        Returns:
            bounds: [(lower, upper), ...] list
        """
        bounds = []
        
        # Measurement model parameters
        # - Factor loadings (zeta): [0.1, 10]
        n_indicators = len(self.config.measurement.indicators)
        bounds.extend([(0.1, 10.0)] * n_indicators)
        
        # - Thresholds (tau): [-10, 10]
        n_thresholds = self.config.measurement.n_categories - 1
        for _ in range(n_indicators):
            bounds.extend([(-10.0, 10.0)] * n_thresholds)
        
        # Structural model parameters (gamma): unbounded
        n_sociodem = len(self.config.structural.sociodemographics)
        bounds.extend([(None, None)] * n_sociodem)
        
        # Choice model parameters
        # - Intercept: unbounded
        bounds.append((None, None))
        
        # - Attribute coefficients (beta): unbounded
        n_attributes = len(self.config.choice.choice_attributes)
        bounds.extend([(None, None)] * n_attributes)
        
        # - Latent variable coefficient (lambda): unbounded
        bounds.append((None, None))
        
        # - Sociodemographic coefficients: unbounded
        if self.config.structural.include_in_choice:
            bounds.extend([(None, None)] * n_sociodem)
        
        return bounds
'''

# _unpack_parameters 메서드 앞에 삽입
content = content.replace(
    '    def _unpack_parameters(self, params: np.ndarray,',
    bounds_method + '    def _unpack_parameters(self, params: np.ndarray,'
)

# 파일 저장
with open('src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("파일 수정 완료!")
print("수정 사항:")
print("1. BFGS → L-BFGS-B")
print("2. Parameter bounds 추가")
print("3. Panel Product 구현")
print("4. 수치 안정성 강화")

