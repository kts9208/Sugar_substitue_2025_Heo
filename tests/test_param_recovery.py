"""
파라미터 복원 테스트 (Parameter Recovery Test)

시뮬레이션 데이터로 알려진 파라미터를 정확히 복원하는지 검증합니다.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc
from scipy.optimize import minimize
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ICLVSimulator:
    """ICLV 모델 시뮬레이션 데이터 생성기"""
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_data(self, n_individuals=200, n_choice_tasks=1):
        """
        시뮬레이션 데이터 생성
        
        Parameters:
        -----------
        n_individuals : int
            개인 수
        n_choice_tasks : int
            개인당 선택 과제 수
        """
        logger.info(f"시뮬레이션 데이터 생성: N={n_individuals}, Tasks={n_choice_tasks}")
        
        # 실제 파라미터 (King 2022 스타일)
        true_params = {
            # 선택모델
            'intercept': 0.5,
            'b_bid': -2.0,
            'lambda': 1.5,
            
            # 구조모델 (간소화: 3개 변수만)
            'gamma_Age': 0.3,
            'gamma_Gender': -0.2,
            'gamma_Income': 0.4,
            
            # 측정모델 (3개 지표)
            'zeta_Q13': 1.0,
            'zeta_Q14': 1.2,
            'zeta_Q15': 0.8,
            
            # 임계값 (5점 척도)
            'tau_Q13_1': -1.5,
            'tau_Q13_2': -0.5,
            'tau_Q13_3': 0.5,
            'tau_Q13_4': 1.5,
            
            'tau_Q14_1': -1.5,
            'tau_Q14_2': -0.5,
            'tau_Q14_3': 0.5,
            'tau_Q14_4': 1.5,
            
            'tau_Q15_1': -1.5,
            'tau_Q15_2': -0.5,
            'tau_Q15_3': 0.5,
            'tau_Q15_4': 1.5,
        }
        
        data_list = []
        
        for i in range(n_individuals):
            # 사회인구학적 변수
            age = np.random.normal(45, 15)
            age_std = (age - 45) / 15
            gender = np.random.binomial(1, 0.5)
            income = np.random.binomial(1, 0.6)
            
            # 잠재변수 (구조방정식)
            eta = np.random.normal(0, 1)
            lv = (
                true_params['gamma_Age'] * age_std +
                true_params['gamma_Gender'] * gender +
                true_params['gamma_Income'] * income +
                eta
            )
            
            # 측정지표 (Ordered Probit)
            q13 = self._generate_ordered_response(lv, true_params['zeta_Q13'], 
                                                   [true_params[f'tau_Q13_{j}'] for j in range(1, 5)])
            q14 = self._generate_ordered_response(lv, true_params['zeta_Q14'],
                                                   [true_params[f'tau_Q14_{j}'] for j in range(1, 5)])
            q15 = self._generate_ordered_response(lv, true_params['zeta_Q15'],
                                                   [true_params[f'tau_Q15_{j}'] for j in range(1, 5)])
            
            # 선택 과제
            for t in range(n_choice_tasks):
                bid = np.random.uniform(0.2, 1.5)
                
                # 선택 (Binary Probit)
                V = (
                    true_params['intercept'] +
                    true_params['b_bid'] * bid +
                    true_params['lambda'] * lv
                )
                prob_yes = norm.cdf(V)
                choice = np.random.binomial(1, prob_yes)
                
                data_list.append({
                    'ID': i,
                    'Task': t,
                    'Age': age,
                    'Age_std': age_std,
                    'Gender': gender,
                    'Income': income,
                    'Q13': q13,
                    'Q14': q14,
                    'Q15': q15,
                    'Bid': bid,
                    'Choice': choice,
                    'LV_true': lv  # 검증용
                })
        
        df = pd.DataFrame(data_list)
        
        logger.info(f"데이터 생성 완료: {len(df)} 관측치")
        logger.info(f"선택 비율: {df['Choice'].mean():.3f}")
        logger.info(f"측정지표 평균: Q13={df['Q13'].mean():.2f}, Q14={df['Q14'].mean():.2f}, Q15={df['Q15'].mean():.2f}")
        
        return df, true_params
    
    def _generate_ordered_response(self, lv, zeta, tau):
        """Ordered Probit으로 범주형 응답 생성"""
        # 연속 잠재응답
        y_star = zeta * lv + np.random.normal(0, 1)
        
        # 범주화
        if y_star < tau[0]:
            return 1
        elif y_star < tau[1]:
            return 2
        elif y_star < tau[2]:
            return 3
        elif y_star < tau[3]:
            return 4
        else:
            return 5


class ICLVEstimator:
    """ICLV 모델 추정기 (최적화 버전)"""
    
    def __init__(self, n_draws=100, seed=42):
        self.n_draws = n_draws
        self.seed = seed
        self.halton_draws = None
        self.results = None
    
    def fit(self, data):
        """모델 추정"""
        logger.info("="*70)
        logger.info("ICLV 모델 추정 시작")
        logger.info("="*70)
        
        start_time = time.time()
        
        # Halton draws 생성
        n_individuals = data['ID'].nunique()
        self._generate_halton_draws(n_individuals)
        
        # 초기값 설정
        initial_params = self._get_initial_values()
        
        # 최적화
        logger.info(f"초기 파라미터 수: {len(initial_params)}")
        logger.info(f"Halton draws: {self.n_draws}")
        logger.info("최적화 시작...")
        
        result = minimize(
            fun=self._negative_log_likelihood,
            x0=initial_params,
            args=(data,),
            method='BFGS',
            options={
                'disp': True,
                'maxiter': 50,
                'gtol': 1e-3
            }
        )
        
        elapsed_time = time.time() - start_time
        
        logger.info("="*70)
        logger.info(f"최적화 완료: {elapsed_time:.1f}초")
        logger.info(f"수렴 여부: {result.success}")
        logger.info(f"최종 로그우도: {-result.fun:.2f}")
        logger.info(f"반복 횟수: {result.nit}")
        logger.info("="*70)
        
        self.results = result
        return self._parse_parameters(result.x)
    
    def _generate_halton_draws(self, n_individuals):
        """Halton draws 생성"""
        logger.info(f"Halton draws 생성: {n_individuals} individuals × {self.n_draws} draws")
        
        sampler = qmc.Halton(d=1, scramble=True, seed=self.seed)
        uniform_draws = sampler.random(n=n_individuals * self.n_draws)
        normal_draws = norm.ppf(uniform_draws).flatten()
        
        self.halton_draws = normal_draws.reshape(n_individuals, self.n_draws)
        logger.info(f"Draws 평균: {self.halton_draws.mean():.6f}, 표준편차: {self.halton_draws.std():.6f}")
    
    def _get_initial_values(self):
        """초기값 설정"""
        return np.array([
            # 선택모델
            0.0,   # intercept
            -1.0,  # b_bid
            1.0,   # lambda
            
            # 구조모델
            0.0,   # gamma_Age
            0.0,   # gamma_Gender
            0.0,   # gamma_Income
            
            # 측정모델
            1.0,   # zeta_Q13
            1.0,   # zeta_Q14
            1.0,   # zeta_Q15
            
            # 임계값 Q13
            -1.0, 0.0, 1.0, 2.0,
            
            # 임계값 Q14
            -1.0, 0.0, 1.0, 2.0,
            
            # 임계값 Q15
            -1.0, 0.0, 1.0, 2.0,
        ])
    
    def _negative_log_likelihood(self, params, data):
        """음의 로그우도 (최소화 목적)"""
        try:
            ll = self._log_likelihood(params, data)
            return -ll
        except:
            return 1e10
    
    def _log_likelihood(self, params, data):
        """로그우도 계산"""
        param_dict = self._parse_parameters(params)
        
        total_ll = 0.0
        
        # 개인별 계산
        for individual_id in data['ID'].unique():
            ind_data = data[data['ID'] == individual_id].iloc[0]
            draws = self.halton_draws[individual_id]
            
            ind_likelihood = 0.0
            
            # 시뮬레이션 (Halton draws)
            for eta in draws:
                # 구조방정식
                lv = (
                    param_dict['gamma_Age'] * ind_data['Age_std'] +
                    param_dict['gamma_Gender'] * ind_data['Gender'] +
                    param_dict['gamma_Income'] * ind_data['Income'] +
                    eta
                )
                
                # 측정모델 우도
                ll_q13 = self._ordered_probit_ll(ind_data['Q13'], lv, param_dict['zeta_Q13'], param_dict['tau_Q13'])
                ll_q14 = self._ordered_probit_ll(ind_data['Q14'], lv, param_dict['zeta_Q14'], param_dict['tau_Q14'])
                ll_q15 = self._ordered_probit_ll(ind_data['Q15'], lv, param_dict['zeta_Q15'], param_dict['tau_Q15'])
                
                # 선택모델 우도
                ll_choice = self._binary_probit_ll(ind_data['Choice'], ind_data['Bid'], lv, param_dict)
                
                # 구조모델 우도 (eta ~ N(0,1))
                ll_structural = norm.logpdf(eta, 0, 1)
                
                # 결합 우도
                joint_ll = ll_q13 + ll_q14 + ll_q15 + ll_choice + ll_structural
                ind_likelihood += np.exp(joint_ll)
            
            # 평균
            ind_likelihood /= self.n_draws
            
            # 로그
            if ind_likelihood > 0:
                total_ll += np.log(ind_likelihood)
            else:
                total_ll += -1e10
        
        return total_ll
    
    def _ordered_probit_ll(self, y, lv, zeta, tau):
        """Ordered Probit 로그우도"""
        k = int(y) - 1
        
        if k == 0:
            prob = norm.cdf(tau[0] - zeta * lv)
        elif k == 4:
            prob = 1 - norm.cdf(tau[3] - zeta * lv)
        else:
            prob = norm.cdf(tau[k] - zeta * lv) - norm.cdf(tau[k-1] - zeta * lv)
        
        prob = np.clip(prob, 1e-10, 1-1e-10)
        return np.log(prob)
    
    def _binary_probit_ll(self, choice, bid, lv, params):
        """Binary Probit 로그우도"""
        V = params['intercept'] + params['b_bid'] * bid + params['lambda'] * lv
        prob = norm.cdf(V)
        prob = np.clip(prob, 1e-10, 1-1e-10)
        
        if choice == 1:
            return np.log(prob)
        else:
            return np.log(1 - prob)
    
    def _parse_parameters(self, params):
        """파라미터 벡터를 딕셔너리로 변환"""
        return {
            'intercept': params[0],
            'b_bid': params[1],
            'lambda': params[2],
            'gamma_Age': params[3],
            'gamma_Gender': params[4],
            'gamma_Income': params[5],
            'zeta_Q13': params[6],
            'zeta_Q14': params[7],
            'zeta_Q15': params[8],
            'tau_Q13': params[9:13],
            'tau_Q14': params[13:17],
            'tau_Q15': params[17:21],
        }


def run_parameter_recovery_test():
    """파라미터 복원 테스트 실행"""
    print("\n" + "="*70)
    print("파라미터 복원 테스트 (Parameter Recovery Test)")
    print("="*70)

    # 1. 시뮬레이션 데이터 생성 (작은 샘플로 빠른 테스트)
    simulator = ICLVSimulator(seed=42)
    data, true_params = simulator.generate_data(n_individuals=100, n_choice_tasks=1)

    # 2. 모델 추정 (적은 draws로 빠른 테스트)
    estimator = ICLVEstimator(n_draws=50, seed=42)
    estimated_params = estimator.fit(data)
    
    # 3. 결과 비교
    print("\n" + "="*70)
    print("파라미터 복원 결과")
    print("="*70)
    print(f"{'Parameter':<20} {'True':>10} {'Estimated':>10} {'Bias':>10} {'Bias %':>10}")
    print("-"*70)
    
    for key in ['intercept', 'b_bid', 'lambda', 'gamma_Age', 'gamma_Gender', 'gamma_Income',
                'zeta_Q13', 'zeta_Q14', 'zeta_Q15']:
        true_val = true_params[key]
        est_val = estimated_params[key]
        bias = est_val - true_val
        bias_pct = (bias / true_val * 100) if true_val != 0 else 0
        
        print(f"{key:<20} {true_val:>10.3f} {est_val:>10.3f} {bias:>10.3f} {bias_pct:>9.1f}%")
    
    print("="*70)
    
    return data, true_params, estimated_params


if __name__ == "__main__":
    data, true_params, estimated_params = run_parameter_recovery_test()

