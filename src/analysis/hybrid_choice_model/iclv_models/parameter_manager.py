"""
파라미터 관리 통합 클래스

동시추정과 순차추정에서 파라미터 처리를 통합하여 관리합니다.
효용함수 파라미터 추가/제거가 용이하도록 설계되었습니다.

주요 기능:
1. 파라미터 이름 리스트 생성 (순서 보장)
2. 딕셔너리 ↔ 배열 변환
3. 초기값 생성

Author: Taeseok Kim
Date: 2025-01-17
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional


class ParameterManager:
    """
    파라미터 관리 통합 클래스
    
    역할:
    - 파라미터 이름 리스트 생성 (순서 보장)
    - 딕셔너리 ↔ 배열 변환
    - 초기값 생성
    
    장점:
    - 파라미터 추가/제거 시 한 곳만 수정
    - 순서 불일치 문제 원천 차단
    - 가독성 및 유지보수성 향상
    """
    
    def __init__(self, config):
        """
        초기화
        
        Args:
            config: MultiLatentConfig 또는 ChoiceConfig
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def get_parameter_names(self, measurement_model, structural_model,
                           choice_model, exclude_measurement: bool = False) -> List[str]:
        """
        전체 파라미터 이름 리스트 생성 (순서 보장)

        이 메서드가 파라미터 순서를 결정합니다.
        파라미터 추가/제거 시 이 메서드만 수정하면 됩니다.

        Args:
            measurement_model: 측정모델 객체
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체
            exclude_measurement: True이면 측정모델 파라미터 제외 (고정 시)

        Returns:
            파라미터 이름 리스트
            예: ['zeta_health_concern_0', 'zeta_health_concern_1', ...,
                 'gamma_health_concern_to_perceived_benefit', ...,
                 'intercept', 'beta_sugar_free', 'beta_health_label', 'beta_price',
                 'lambda_health_concern', 'lambda_perceived_benefit', ...]
        """
        names = []

        # 1. 측정모델 파라미터 (exclude_measurement=True이면 제외)
        if not exclude_measurement:
            names.extend(self._get_measurement_param_names(measurement_model))

        # 2. 구조모델 파라미터
        names.extend(self._get_structural_param_names(structural_model))

        # 3. 선택모델 파라미터
        names.extend(self._get_choice_param_names(choice_model))

        if exclude_measurement:
            self.logger.info(f"파라미터 이름 리스트 생성 완료: {len(names)}개 (측정모델 제외)")
        else:
            self.logger.info(f"파라미터 이름 리스트 생성 완료: {len(names)}개")

        return names
    
    def get_measurement_param_names(self, measurement_model) -> List[str]:
        """
        측정모델 파라미터 이름만 반환 (public 메서드)

        Args:
            measurement_model: 측정모델 객체

        Returns:
            측정모델 파라미터 이름 리스트
        """
        return self._get_measurement_param_names(measurement_model)

    def get_optimized_parameter_names(self, structural_model, choice_model) -> List[str]:
        """
        동시추정용: 구조모델 + 선택모델 파라미터 이름만 반환

        측정모델 파라미터는 항상 제외됩니다 (CFA 결과로 고정).

        Args:
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체

        Returns:
            최적화 파라미터 이름 리스트 (측정모델 제외)
        """
        names = []

        # 구조모델 파라미터
        names.extend(self._get_structural_param_names(structural_model))

        # 선택모델 파라미터
        names.extend(self._get_choice_param_names(choice_model))

        self.logger.info(f"최적화 파라미터 이름 리스트 생성 완료: {len(names)}개 (측정모델 제외)")

        return names

    def _get_measurement_param_names(self, measurement_model) -> List[str]:
        """측정모델 파라미터 이름 생성"""
        names = []

        if hasattr(measurement_model, 'models'):
            # 다중 잠재변수
            for lv_name, model in measurement_model.models.items():
                indicators = model.config.indicators

                # zeta (요인적재량)
                # ✅ indicator 이름 사용 (initializer.py와 동일)
                for indicator in indicators:
                    names.append(f'zeta_{lv_name}_{indicator}')

                # sigma_sq (연속형) 또는 tau (순서형)
                if hasattr(model.config, 'measurement_method'):
                    if model.config.measurement_method == 'continuous_linear':
                        # ✅ indicator 이름 사용
                        for indicator in indicators:
                            names.append(f'sigma_sq_{lv_name}_{indicator}')
                    else:
                        # ordered_probit
                        n_thresholds = model.config.n_categories - 1
                        # ✅ indicator 이름 사용
                        for indicator in indicators:
                            for k in range(n_thresholds):
                                names.append(f'tau_{lv_name}_{indicator}_{k+1}')

        return names
    
    def _get_structural_param_names(self, structural_model) -> List[str]:
        """구조모델 파라미터 이름 생성"""
        names = []
        
        # 계층적 구조
        if hasattr(structural_model, 'is_hierarchical') and structural_model.is_hierarchical:
            for path in structural_model.hierarchical_paths:
                predictor = path['predictors'][0]  # 단일 predictor 가정
                target = path['target']
                names.append(f'gamma_{predictor}_to_{target}')
        
        return names
    
    def _get_choice_param_names(self, choice_model) -> List[str]:
        """
        선택모델 파라미터 이름 생성 (대안별 모델 지원)

        ✅ 순차추정과 동일한 파라미터 구조:
        - ASC (Alternative Specific Constants): asc_sugar, asc_sugar_free
        - Beta (속성 계수): beta_health_label, beta_price (sugar_free 제외)
        - Theta (대안별 LV 계수): theta_sugar_*, theta_sugar_free_*
        - Gamma (대안별 상호작용): gamma_sugar_*_*, gamma_sugar_free_*_*
        """
        names = []

        # ✅ 대안별 모델 (Multinomial Logit with ASC)
        if hasattr(choice_model, 'n_alternatives') and choice_model.n_alternatives == 3:
            # ASC (Alternative Specific Constants)
            # opt-out은 reference이므로 파라미터 없음
            names.append('asc_sugar')       # 일반당 상수
            names.append('asc_sugar_free')  # 무설탕 상수

            # Beta (속성 계수) - sugar_free 제외
            for attr in choice_model.config.choice_attributes:
                if attr != 'sugar_free':  # sugar_free는 ASC에 포함됨
                    names.append(f'beta_{attr}')

            # Theta (대안별 잠재변수 계수)
            for lv_name in choice_model.main_lvs:
                names.append(f'theta_sugar_{lv_name}')
                names.append(f'theta_sugar_free_{lv_name}')

            # Gamma (대안별 LV-Attribute 상호작용)
            for interaction in choice_model.lv_attribute_interactions:
                lv_name = interaction['lv']
                attr_name = interaction['attribute']
                names.append(f'gamma_sugar_{lv_name}_{attr_name}')
                names.append(f'gamma_sugar_free_{lv_name}_{attr_name}')

        # ✅ Binary/기타 모델 (하위 호환)
        else:
            # beta_intercept
            names.append('beta_intercept')

            # beta (속성 계수)
            for attr in choice_model.config.choice_attributes:
                names.append(f'beta_{attr}')

            # lambda (잠재변수 계수)
            for lv_name in choice_model.main_lvs:
                names.append(f'lambda_{lv_name}')

            # gamma (LV-Attribute 상호작용)
            for interaction in choice_model.lv_attribute_interactions:
                lv_name = interaction['lv']
                attr_name = interaction['attribute']
                names.append(f'gamma_{lv_name}_{attr_name}')

        return names

    def array_to_dict_optimized(self, param_array: np.ndarray, param_names: List[str],
                                measurement_model, structural_model, choice_model) -> Dict:
        """
        동시추정 전용: 최적화 파라미터 배열을 딕셔너리로 변환

        ✅ 측정모델 파라미터는 처리하지 않음
        ✅ 구조모델 + 선택모델만 변환

        Args:
            param_array: 최적화 파라미터 배열 (8개, 측정모델 제외)
            param_names: 최적화 파라미터 이름 리스트
            measurement_model: 측정모델 (측정모델 파라미터 추출용)
            structural_model: 구조모델
            choice_model: 선택모델

        Returns:
            파라미터 딕셔너리
            {
                'measurement': {lv_name: {'zeta': array, 'sigma_sq': array}},  # CFA 결과
                'structural': {param_name: float},
                'choice': {param_name: float, ...}
            }
        """
        param_dict = {
            'measurement': {},
            'structural': {},
            'choice': {}
        }

        # ✅ 측정모델 파라미터: measurement_model 객체에서 직접 추출 (CFA 결과)
        if hasattr(measurement_model, 'models'):
            for lv_name, model in measurement_model.models.items():
                # ✅ 모델 객체에서 직접 파라미터 추출 (config가 아님!)
                # test_gpu_batch_iclv.py에서 model.config.zeta/sigma_sq/alpha로 설정했으므로
                # config에서 추출
                if hasattr(model, 'config'):
                    zeta = getattr(model.config, 'zeta', np.array([]))
                    sigma_sq = getattr(model.config, 'sigma_sq', np.array([]))
                    alpha = getattr(model.config, 'alpha', None)  # ✅ 절편 추가
                else:
                    # config가 없으면 모델 객체에서 직접
                    zeta = getattr(model, 'zeta', np.array([]))
                    sigma_sq = getattr(model, 'sigma_sq', np.array([]))
                    alpha = getattr(model, 'alpha', None)  # ✅ 절편 추가

                # 디버깅: 추출된 파라미터 로깅
                if len(zeta) == 0 or len(sigma_sq) == 0:
                    self.logger.warning(f"[array_to_dict_optimized] {lv_name}: zeta={len(zeta)}개, sigma_sq={len(sigma_sq)}개 (빈 배열!)")
                else:
                    alpha_info = f", alpha={len(alpha)}개" if alpha is not None else ", alpha=None"
                    self.logger.debug(f"[array_to_dict_optimized] {lv_name}: zeta={len(zeta)}개, sigma_sq={len(sigma_sq)}개{alpha_info}")

                param_dict['measurement'][lv_name] = {
                    'zeta': zeta,
                    'sigma_sq': sigma_sq,
                    'alpha': alpha  # ✅ 절편 추가
                }

        # ✅ 구조모델 + 선택모델 파라미터: 배열에서 추출
        for i, name in enumerate(param_names):
            value = param_array[i]

            if name.startswith('gamma_') and '_to_' in name:
                # 구조모델
                param_dict['structural'][name] = value

            elif name.startswith('asc_'):
                # ASC
                param_dict['choice'][name] = value

            elif name.startswith('theta_'):
                # Theta
                param_dict['choice'][name] = value

            elif name.startswith('beta_'):
                # Beta (개별 키로 저장)
                param_dict['choice'][name] = value

            elif name.startswith('gamma_') and ('_sugar_' in name or '_sugar_free_' in name):
                # Gamma 상호작용
                param_dict['choice'][name] = value

            elif name.startswith('lambda_'):
                # Lambda
                param_dict['choice'][name] = value

        return param_dict

    def array_to_dict(self, param_array: np.ndarray, param_names: List[str],
                     measurement_model, structural_model, choice_model) -> Dict:
        """
        파라미터 배열을 딕셔너리로 변환

        Args:
            param_array: 파라미터 배열
            param_names: 파라미터 이름 리스트 (get_parameter_names()로 생성)
            measurement_model: 측정모델 객체
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체

        Returns:
            파라미터 딕셔너리
            {
                'measurement': {lv_name: {'zeta': array, 'sigma_sq': array or 'tau': array}},
                'structural': {param_name: float},
                'choice': {'intercept': float, 'beta': array, 'lambda_*': float, ...}
            }
        """
        param_dict = {
            'measurement': {},
            'structural': {},
            'choice': {}
        }

        # 측정모델 파라미터 초기화
        if hasattr(measurement_model, 'models'):
            for lv_name, model in measurement_model.models.items():
                n_indicators = len(model.config.indicators)
                param_dict['measurement'][lv_name] = {
                    'zeta': np.zeros(n_indicators)
                }

                if hasattr(model.config, 'measurement_method'):
                    if model.config.measurement_method == 'continuous_linear':
                        param_dict['measurement'][lv_name]['sigma_sq'] = np.zeros(n_indicators)
                    else:
                        n_thresholds = model.config.n_categories - 1
                        param_dict['measurement'][lv_name]['tau'] = np.zeros((n_indicators, n_thresholds))

        # 선택모델 beta 초기화
        n_attributes = len(choice_model.config.choice_attributes)
        param_dict['choice']['beta'] = np.zeros(n_attributes)

        # 파라미터 이름 기반으로 값 할당
        for name, value in zip(param_names, param_array):
            if name.startswith('zeta_'):
                # zeta_health_concern_q6 → lv_name='health_concern', indicator='q6'
                parts = name.split('_')
                lv_name = '_'.join(parts[1:-1])
                indicator = parts[-1]

                # indicator 이름으로 인덱스 찾기
                model = measurement_model.models[lv_name]
                indicators = model.config.indicators
                idx = indicators.index(indicator)
                param_dict['measurement'][lv_name]['zeta'][idx] = value

            elif name.startswith('sigma_sq_'):
                # sigma_sq_health_concern_q6
                parts = name.split('_')
                lv_name = '_'.join(parts[2:-1])
                indicator = parts[-1]

                # indicator 이름으로 인덱스 찾기
                model = measurement_model.models[lv_name]
                indicators = model.config.indicators
                idx = indicators.index(indicator)
                param_dict['measurement'][lv_name]['sigma_sq'][idx] = value

            elif name.startswith('tau_'):
                # tau_health_concern_q6_1 → lv_name='health_concern', indicator='q6', tau_idx=1
                parts = name.split('_')
                lv_name = '_'.join(parts[1:-2])
                indicator = parts[-2]
                tau_idx = int(parts[-1]) - 1  # 1-based → 0-based

                # indicator 이름으로 인덱스 찾기
                model = measurement_model.models[lv_name]
                indicators = model.config.indicators
                ind_idx = indicators.index(indicator)
                param_dict['measurement'][lv_name]['tau'][ind_idx][tau_idx] = value

            elif name.startswith('gamma_') and '_to_' in name:
                # gamma_health_concern_to_perceived_benefit (구조모델)
                param_dict['structural'][name] = value

            # ✅ 대안별 모델 파라미터
            elif name.startswith('asc_'):
                # asc_sugar, asc_sugar_free
                param_dict['choice'][name] = value

            elif name.startswith('theta_'):
                # theta_sugar_purchase_intention, theta_sugar_free_nutrition_knowledge
                param_dict['choice'][name] = value

            elif name.startswith('gamma_sugar_') or name.startswith('gamma_sugar_free_'):
                # gamma_sugar_purchase_intention_health_label
                # gamma_sugar_free_nutrition_knowledge_price
                param_dict['choice'][name] = value

            # ✅ Binary/기타 모델 파라미터 (하위 호환)
            elif name.startswith('gamma_') and not '_to_' in name:
                # gamma_purchase_intention_health_label (LV-Attribute 상호작용)
                param_dict['choice'][name] = value

            elif name == 'beta_intercept':
                # beta_intercept
                param_dict['choice']['intercept'] = value

            elif name.startswith('beta_'):
                # beta_health_label, beta_price
                attr_name = name.replace('beta_', '')
                # ✅ 대안별 모델에서는 sugar_free가 없으므로 인덱스 조정 필요
                if attr_name in choice_model.config.choice_attributes:
                    attr_idx = choice_model.config.choice_attributes.index(attr_name)
                    param_dict['choice']['beta'][attr_idx] = value
                else:
                    # 속성이 리스트에 없으면 딕셔너리에 직접 저장
                    param_dict['choice'][name] = value

            elif name.startswith('lambda_'):
                # lambda_purchase_intention, lambda_nutrition_knowledge
                param_dict['choice'][name] = value

        return param_dict

    def dict_to_array_optimized(self, param_dict: Dict, param_names: List[str],
                                structural_model, choice_model) -> np.ndarray:
        """
        동시추정 전용: 구조모델 + 선택모델만 배열로 변환

        ✅ 단순화된 로직: 파라미터 이름 순서대로 값을 추출
        ✅ 구조모델 파라미터는 'structural'에서, 나머지는 'choice'에서 추출

        Args:
            param_dict: 파라미터 딕셔너리
                {'structural': {...}, 'choice': {...}}
            param_names: 최적화 파라미터 이름 리스트 (get_optimized_parameter_names()로 생성)
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체

        Returns:
            파라미터 배열 (측정모델 제외)
        """
        param_array = []

        for i, name in enumerate(param_names):
            # ✅ 단순화된 로직: gamma_*_to_*는 structural, 나머지는 choice
            if name.startswith('gamma_') and '_to_' in name:
                # 구조모델 파라미터
                if name in param_dict['structural']:
                    value = param_dict['structural'][name]
                else:
                    self.logger.warning(f"구조모델 파라미터 '{name}'를 찾을 수 없음, 0.1 사용")
                    value = 0.1
            else:
                # 선택모델 파라미터 (asc, beta, theta, gamma, lambda 등)
                if name in param_dict['choice']:
                    value = param_dict['choice'][name]
                else:
                    self.logger.warning(f"선택모델 파라미터 '{name}'를 찾을 수 없음, 0.1 사용")
                    value = 0.1

            param_array.append(value)

        return np.array(param_array)

    def dict_to_array(self, param_dict: Dict, param_names: List[str], measurement_model=None) -> np.ndarray:
        """
        파라미터 딕셔너리를 배열로 변환

        Args:
            param_dict: 파라미터 딕셔너리
                {'measurement': {...}, 'structural': {...}, 'choice': {...}}
            param_names: 파라미터 이름 리스트 (get_parameter_names()로 생성)
            measurement_model: 측정모델 (reference indicator 확인용, gradient 변환 시 필요)

        Returns:
            파라미터 배열
        """
        param_array = []

        for name in param_names:
            if name.startswith('zeta_'):
                # ✅ indicator 이름 파싱 (예: zeta_health_concern_q7)
                # parts = ['zeta', 'health', 'concern', 'q7']
                # lv_name = 'health_concern', indicator = 'q7'
                parts = name.split('_')
                # 마지막 부분이 indicator 이름 (q7, q8, ...)
                indicator = parts[-1]
                # 중간 부분이 lv_name (health_concern, perceived_benefit, ...)
                lv_name = '_'.join(parts[1:-1])

                # zeta는 배열 형태 (indicator 순서대로)
                zeta_array = param_dict['measurement'][lv_name]['zeta']
                if isinstance(zeta_array, np.ndarray):
                    # ✅ Reference indicator 처리
                    # param_names에서 현재 lv의 zeta 파라미터들의 순서를 찾음
                    lv_zeta_names = [n for n in param_names if n.startswith(f'zeta_{lv_name}_')]
                    idx = lv_zeta_names.index(name)

                    # fix_first_loading 확인 (gradient 변환 시에만 필요)
                    fix_first_loading = False
                    if measurement_model is not None:
                        if hasattr(measurement_model, 'models') and lv_name in measurement_model.models:
                            config = measurement_model.models[lv_name].config
                            fix_first_loading = getattr(config, 'fix_first_loading', True)

                    # Reference indicator (첫 번째)는 gradient = 0
                    if fix_first_loading and idx == 0:
                        param_array.append(0.0)
                    else:
                        # fix_first_loading=True이면 배열에서 첫 번째 제외되어 있음
                        actual_idx = idx - 1 if fix_first_loading else idx
                        param_array.append(zeta_array[actual_idx])
                else:
                    param_array.append(zeta_array)

            elif name.startswith('sigma_sq_'):
                # ✅ indicator 이름 파싱 (예: sigma_sq_health_concern_q7)
                parts = name.split('_')
                indicator = parts[-1]
                lv_name = '_'.join(parts[2:-1])  # 'sigma_sq' 제외

                # sigma_sq는 배열 형태
                sigma_sq_array = param_dict['measurement'][lv_name]['sigma_sq']
                if isinstance(sigma_sq_array, np.ndarray):
                    lv_sigma_sq_names = [n for n in param_names if n.startswith(f'sigma_sq_{lv_name}_')]
                    idx = lv_sigma_sq_names.index(name)
                    param_array.append(sigma_sq_array[idx])
                else:
                    param_array.append(sigma_sq_array)

            elif name.startswith('tau_'):
                # ✅ indicator 이름 파싱 (예: tau_health_concern_q7_1)
                parts = name.split('_')
                tau_idx = int(parts[-1]) - 1  # k+1 형식이므로 -1
                indicator = parts[-2]
                lv_name = '_'.join(parts[1:-2])  # 'tau' 제외, indicator와 tau_idx 제외

                # tau는 2D 배열 형태
                tau_array = param_dict['measurement'][lv_name]['tau']
                lv_tau_names = [n for n in param_names if n.startswith(f'tau_{lv_name}_') and n.endswith(f'_{parts[-1]}')]
                # 같은 threshold의 tau들 중에서 현재 indicator의 인덱스
                lv_tau_base_names = [n for n in param_names if n.startswith(f'tau_{lv_name}_') and not n.endswith(f'_{parts[-1]}')]
                # indicator별로 그룹화
                indicators_in_order = []
                for n in param_names:
                    if n.startswith(f'tau_{lv_name}_'):
                        ind = '_'.join(n.split('_')[2:-1])
                        if ind not in indicators_in_order:
                            indicators_in_order.append(ind)
                ind_idx = indicators_in_order.index(indicator)
                param_array.append(tau_array[ind_idx][tau_idx])

            elif name.startswith('gamma_') and '_to_' in name:
                # 구조모델 파라미터
                param_array.append(param_dict['structural'][name])

            # ✅ 대안별 모델 파라미터
            elif name.startswith('asc_'):
                # asc_sugar, asc_sugar_free
                param_array.append(param_dict['choice'].get(name, 0.0))

            elif name.startswith('theta_'):
                # theta_sugar_purchase_intention, theta_sugar_free_nutrition_knowledge
                param_array.append(param_dict['choice'].get(name, 0.0))

            elif name.startswith('gamma_sugar_') or name.startswith('gamma_sugar_free_'):
                # gamma_sugar_purchase_intention_health_label
                # gamma_sugar_free_nutrition_knowledge_price
                param_array.append(param_dict['choice'].get(name, 0.0))

            # ✅ Binary/기타 모델 파라미터 (하위 호환)
            elif name.startswith('gamma_') and not '_to_' in name:
                # LV-Attribute 상호작용 파라미터
                param_array.append(param_dict['choice'].get(name, 0.0))

            elif name == 'beta_intercept':
                # beta_intercept
                param_array.append(param_dict['choice'].get('intercept', 0.0))

            elif name.startswith('beta_'):
                # ✅ 먼저 flat 딕셔너리 형식 확인 (CSV에서 로드한 경우)
                if name in param_dict['choice']:
                    # 딕셔너리에 직접 저장된 경우 (예: beta_health_label: 1.0)
                    param_array.append(param_dict['choice'][name])
                elif 'beta' in param_dict['choice']:
                    # beta는 배열 형태이므로 현재까지 처리한 beta 개수로 인덱스 결정
                    # ✅ beta_intercept 제외하고 카운트
                    beta_count = sum(1 for n in param_names[:param_names.index(name)]
                                    if n.startswith('beta_') and n != 'beta_intercept')
                    beta_array = param_dict['choice']['beta']
                    if isinstance(beta_array, np.ndarray) and beta_count < len(beta_array):
                        param_array.append(beta_array[beta_count])
                    else:
                        param_array.append(beta_array)
                else:
                    # 둘 다 없으면 0.0으로 초기화
                    param_array.append(0.0)

            elif name.startswith('lambda_'):
                param_array.append(param_dict['choice'].get(name, 0.0))

        return np.array(param_array)

    def get_initial_values(self, param_names: List[str],
                          measurement_model, structural_model, choice_model) -> np.ndarray:
        """
        초기값 생성

        Args:
            param_names: 파라미터 이름 리스트 (get_parameter_names()로 생성)
            measurement_model: 측정모델 객체
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체

        Returns:
            초기값 배열
        """
        initial_values = []

        for name in param_names:
            if name.startswith('zeta_'):
                # 요인적재량: 1.0
                initial_values.append(1.0)

            elif name.startswith('sigma_sq_'):
                # 오차분산: 1.0
                initial_values.append(1.0)

            elif name.startswith('tau_'):
                # 임계값: 등간격으로 초기화
                parts = name.split('_')
                lv_name = '_'.join(parts[1:-2])
                tau_idx = int(parts[-1])

                # 해당 LV의 n_thresholds 찾기
                if hasattr(measurement_model, 'models'):
                    model = measurement_model.models[lv_name]
                    n_thresholds = model.config.n_categories - 1
                    tau_init = np.linspace(-2, 2, n_thresholds)[tau_idx]
                    initial_values.append(tau_init)

            elif name.startswith('gamma_') and '_to_' in name:
                # 구조모델 계수: 0.5
                initial_values.append(0.5)

            # ✅ 대안별 모델 파라미터
            elif name.startswith('asc_'):
                # ASC (Alternative Specific Constants): 0.5
                initial_values.append(0.5)

            elif name.startswith('theta_'):
                # Theta (대안별 LV 계수): 0.5
                initial_values.append(0.5)

            elif name.startswith('gamma_sugar_') or name.startswith('gamma_sugar_free_'):
                # Gamma (대안별 상호작용): 0.1
                initial_values.append(0.1)

            # ✅ Binary/기타 모델 파라미터 (하위 호환)
            elif name.startswith('gamma_') and not '_to_' in name:
                # LV-Attribute 상호작용: 0.5
                initial_values.append(0.5)

            elif name == 'beta_intercept':
                # 절편: 0.1
                initial_values.append(0.1)

            elif name.startswith('beta_'):
                # 속성 계수
                attr_name = name.replace('beta_', '')
                if 'price' in attr_name.lower():
                    # 가격: 음수 초기값
                    initial_values.append(-1.0)
                else:
                    # 기타 속성: 작은 양수
                    initial_values.append(0.1)

            elif name.startswith('lambda_'):
                # 잠재변수 계수: 1.0
                initial_values.append(1.0)

        return np.array(initial_values)

    def split_measurement_params(self, params_full: np.ndarray,
                                 measurement_model, structural_model,
                                 choice_model) -> tuple:
        """
        전체 파라미터 배열을 측정모델 파라미터와 나머지로 분리

        Args:
            params_full: 전체 파라미터 배열
            measurement_model: 측정모델 객체
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체

        Returns:
            (measurement_params, other_params) 튜플
        """
        measurement_names = self._get_measurement_param_names(measurement_model)
        n_measurement = len(measurement_names)

        measurement_params = params_full[:n_measurement]
        other_params = params_full[n_measurement:]

        return measurement_params, other_params

    def combine_params(self, measurement_params: np.ndarray,
                      other_params: np.ndarray) -> np.ndarray:
        """
        측정모델 파라미터와 나머지 파라미터를 결합

        Args:
            measurement_params: 측정모델 파라미터 배열
            other_params: 구조모델 + 선택모델 파라미터 배열

        Returns:
            전체 파라미터 배열
        """
        return np.concatenate([measurement_params, other_params])

    def get_optimized_parameter_bounds(self, structural_model, choice_model) -> List[tuple]:
        """
        동시추정용: 구조모델 + 선택모델 파라미터 bounds만 반환

        측정모델 파라미터는 항상 제외됩니다 (CFA 결과로 고정).

        Args:
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체

        Returns:
            최적화 파라미터 bounds 리스트 (측정모델 제외)
        """
        param_names = self.get_optimized_parameter_names(structural_model, choice_model)

        bounds = []
        for name in param_names:
            bounds.append(self._get_bound_for_param(name))

        self.logger.info(f"최적화 파라미터 bounds 생성 완료: {len(bounds)}개 (측정모델 제외)")

        return bounds

    def get_parameter_bounds(self, measurement_model, structural_model,
                            choice_model, exclude_measurement: bool = False) -> List[tuple]:
        """
        전체 파라미터 bounds 리스트 생성 (L-BFGS-B용)

        ✅ 파라미터 이름 순서와 동일한 순서로 bounds 생성
        ✅ 파라미터 구조 변경 시 이 메서드만 수정하면 됨
        ✅ Optimizer 종류와 무관하게 동일한 로직 사용
        ✅ exclude_measurement=True이면 측정모델 파라미터 제외

        Args:
            measurement_model: 측정모델 객체
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체
            exclude_measurement: True이면 측정모델 파라미터 제외

        Returns:
            bounds 리스트: [(lower, upper), ...]
            예: [(0.1, 10.0), (0.01, 100.0), ..., (None, None), ...]
        """
        param_names = self.get_parameter_names(
            measurement_model, structural_model, choice_model,
            exclude_measurement=exclude_measurement
        )

        bounds = []
        for name in param_names:
            bounds.append(self._get_bound_for_param(name))

        if exclude_measurement:
            self.logger.info(f"파라미터 bounds 생성 완료: {len(bounds)}개 (측정모델 제외)")
        else:
            self.logger.info(f"파라미터 bounds 생성 완료: {len(bounds)}개")

        return bounds

    def _get_bound_for_param(self, param_name: str) -> tuple:
        """
        파라미터 이름 기반으로 적절한 bound 반환

        Args:
            param_name: 파라미터 이름

        Returns:
            (lower, upper) 튜플
            - None은 unbounded를 의미
        """
        # 측정모델 파라미터
        if param_name.startswith('zeta_'):
            # Factor loadings: [-2.0, 2.0]
            # 더 엄격한 제약으로 수치 안정성 향상
            return (-2.0, 2.0)

        elif param_name.startswith('sigma_sq_'):
            # Measurement error variance: [-15.0, 15.0]
            # 확대된 범위로 측정 오차 분산 추정 허용
            return (-15.0, 15.0)

        elif param_name.startswith('tau_'):
            # Thresholds (순서형): [-10.0, 10.0]
            # 너무 크면 확률이 0 또는 1에 가까워져 수치 문제
            return (-10.0, 10.0)

        # 구조모델 파라미터
        elif param_name.startswith('gamma_') and '_to_' in param_name:
            # 구조방정식 계수: [-2.0, 2.0]
            # 경로계수에 합리적 제약 설정
            return (-2.0, 2.0)

        # 선택모델 파라미터
        elif param_name.startswith('asc_'):
            # Alternative-specific constants: unbounded
            return (None, None)

        elif param_name == 'intercept':
            # Binary logit intercept: unbounded
            return (None, None)

        elif param_name.startswith('beta_'):
            # Attribute coefficients: [-10.0, 10.0]
            # 확대된 범위로 속성 계수 추정 허용
            return (-10.0, 10.0)

        elif param_name.startswith('theta_'):
            # LV main effects (multinomial): unbounded
            return (None, None)

        elif param_name.startswith('lambda_'):
            # LV coefficients (binary): unbounded
            return (None, None)

        elif param_name.startswith('gamma_') and '_to_' not in param_name:
            # LV-Attribute interactions: unbounded
            return (None, None)

        else:
            # 기타 파라미터: unbounded (안전한 기본값)
            self.logger.warning(f"알 수 없는 파라미터 타입: {param_name}, unbounded로 설정")
            return (None, None)

