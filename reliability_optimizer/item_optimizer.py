"""
신뢰도 최적화 모듈 - 문항 제거를 통한 신뢰도 개선

이 모듈은 기존 신뢰도 분석 결과를 입력받아 AVE 기준을 만족하지 못하는 요인의
문항들을 체계적으로 제거하여 크론바흐 알파, CR, AVE 기준을 모두 만족하는
최적의 문항 조합을 찾는 기능을 제공합니다.

Author: Reliability Optimization System
Date: 2025-01-02
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from itertools import combinations
import warnings
from pathlib import Path
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReliabilityOptimizer:
    """
    신뢰도 최적화를 위한 문항 제거 분석기
    
    기존 신뢰도 분석 결과를 바탕으로 AVE 기준을 만족하지 못하는 요인의
    문항들을 체계적으로 제거하여 최적의 문항 조합을 찾습니다.
    """
    
    # 신뢰도 기준값
    RELIABILITY_THRESHOLDS = {
        'cronbach_alpha': 0.7,
        'composite_reliability': 0.7,
        'ave': 0.5,
        'min_items': 3  # 최소 문항 수
    }
    
    def __init__(self, reliability_results_dir: str = "reliability_analysis_results"):
        """
        초기화
        
        Args:
            reliability_results_dir (str): 기존 신뢰도 분석 결과 디렉토리
        """
        self.results_dir = Path(reliability_results_dir)
        self.reliability_summary = None
        self.factor_loadings = None
        self.raw_data = None
        self.optimization_results = {}
        
        logger.info(f"신뢰도 최적화기 초기화 완료: {self.results_dir}")
    
    def load_reliability_results(self) -> bool:
        """
        기존 신뢰도 분석 결과 로드
        
        Returns:
            bool: 로드 성공 여부
        """
        try:
            # 신뢰도 요약 결과 로드
            summary_path = self.results_dir / "reliability_summary.csv"
            if summary_path.exists():
                self.reliability_summary = pd.read_csv(summary_path)
                logger.info(f"신뢰도 요약 결과 로드 완료: {len(self.reliability_summary)} 요인")
            else:
                logger.error(f"신뢰도 요약 파일을 찾을 수 없습니다: {summary_path}")
                return False
            
            # 요인부하량 결과 로드
            loadings_path = self.results_dir / "factor_loadings.csv"
            if loadings_path.exists():
                self.factor_loadings = pd.read_csv(loadings_path)
                logger.info(f"요인부하량 결과 로드 완료: {len(self.factor_loadings)} 문항")
            else:
                logger.error(f"요인부하량 파일을 찾을 수 없습니다: {loadings_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"신뢰도 결과 로드 중 오류: {e}")
            return False
    
    def load_raw_data(self, data_path: str) -> bool:
        """
        원시 데이터 로드 (크론바흐 알파 계산용)
        
        Args:
            data_path (str): 원시 데이터 파일 경로
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            if data_path.endswith('.csv'):
                self.raw_data = pd.read_csv(data_path, encoding='utf-8-sig')
            elif data_path.endswith('.xlsx'):
                self.raw_data = pd.read_excel(data_path)
            else:
                logger.error(f"지원하지 않는 파일 형식: {data_path}")
                return False
            
            logger.info(f"원시 데이터 로드 완료: {self.raw_data.shape}")
            return True
            
        except Exception as e:
            logger.error(f"원시 데이터 로드 중 오류: {e}")
            return False
    
    def identify_problematic_factors(self) -> List[str]:
        """
        AVE 기준을 만족하지 못하는 요인들 식별
        
        Returns:
            List[str]: 문제가 있는 요인명 리스트
        """
        if self.reliability_summary is None:
            logger.error("신뢰도 요약 결과가 로드되지 않았습니다.")
            return []
        
        problematic_factors = []
        
        for _, row in self.reliability_summary.iterrows():
            factor_name = row['Factor']
            ave = row['AVE']
            ave_acceptable = row['AVE_Acceptable']
            
            if not ave_acceptable or ave < self.RELIABILITY_THRESHOLDS['ave']:
                problematic_factors.append(factor_name)
                logger.info(f"문제 요인 발견: {factor_name} (AVE: {ave:.4f})")
        
        return problematic_factors
    
    def get_factor_items(self, factor_name: str) -> List[str]:
        """
        특정 요인의 문항들 추출
        
        Args:
            factor_name (str): 요인명
            
        Returns:
            List[str]: 문항 리스트
        """
        if self.factor_loadings is None:
            logger.error("요인부하량 결과가 로드되지 않았습니다.")
            return []
        
        factor_items = self.factor_loadings[
            self.factor_loadings['Factor'] == factor_name
        ]['Item'].tolist()
        
        return factor_items
    
    def calculate_cronbach_alpha(self, items: List[str]) -> float:
        """
        주어진 문항들의 크론바흐 알파 계산
        
        Args:
            items (List[str]): 문항 리스트
            
        Returns:
            float: 크론바흐 알파 값
        """
        if self.raw_data is None:
            logger.error("원시 데이터가 로드되지 않았습니다.")
            return np.nan
        
        try:
            # 해당 문항들만 추출
            item_data = self.raw_data[items].dropna()
            
            if len(item_data) == 0:
                return np.nan
            
            # 문항 수
            k = len(items)
            
            if k < 2:
                return np.nan
            
            # 각 문항의 분산
            item_variances = item_data.var(ddof=1)
            sum_item_var = item_variances.sum()
            
            # 전체 점수의 분산
            total_scores = item_data.sum(axis=1)
            total_var = total_scores.var(ddof=1)
            
            # 크론바흐 알파 계산
            if total_var == 0:
                return np.nan
            
            alpha = (k / (k - 1)) * (1 - sum_item_var / total_var)
            return alpha
            
        except Exception as e:
            logger.error(f"크론바흐 알파 계산 중 오류: {e}")
            return np.nan
    
    def calculate_cr_and_ave(self, factor_name: str, items: List[str]) -> Tuple[float, float]:
        """
        주어진 문항들의 CR과 AVE 계산
        
        Args:
            factor_name (str): 요인명
            items (List[str]): 문항 리스트
            
        Returns:
            Tuple[float, float]: (CR, AVE) 값
        """
        if self.factor_loadings is None:
            logger.error("요인부하량 결과가 로드되지 않았습니다.")
            return np.nan, np.nan
        
        try:
            # 해당 문항들의 요인부하량 추출
            factor_data = self.factor_loadings[
                (self.factor_loadings['Factor'] == factor_name) &
                (self.factor_loadings['Item'].isin(items))
            ]
            
            if len(factor_data) == 0:
                return np.nan, np.nan
            
            # 표준화된 요인부하량 (Loading 컬럼 사용)
            loadings = factor_data['Loading'].values
            
            # 오차분산 계산 (1 - λ²)
            error_variances = 1 - (loadings ** 2)
            
            # CR 계산: (Σλ)² / [(Σλ)² + Σδ]
            sum_loadings = np.sum(loadings)
            sum_loadings_squared = np.sum(loadings ** 2)
            sum_error_var = np.sum(error_variances)
            
            numerator = sum_loadings ** 2
            denominator = numerator + sum_error_var
            
            if denominator == 0:
                cr = np.nan
            else:
                cr = numerator / denominator
            
            # AVE 계산: Σλ² / (Σλ² + Σδ)
            ave_denominator = sum_loadings_squared + sum_error_var
            
            if ave_denominator == 0:
                ave = np.nan
            else:
                ave = sum_loadings_squared / ave_denominator
            
            return cr, ave

        except Exception as e:
            logger.error(f"CR/AVE 계산 중 오류: {e}")
            return np.nan, np.nan

    def optimize_factor_reliability(self, factor_name: str, max_removals: int = 10) -> Dict[str, Any]:
        """
        특정 요인의 신뢰도 최적화

        Args:
            factor_name (str): 최적화할 요인명
            max_removals (int): 최대 제거할 문항 수

        Returns:
            Dict[str, Any]: 최적화 결과
        """
        logger.info(f"요인 '{factor_name}' 신뢰도 최적화 시작")

        # 현재 요인의 문항들 가져오기
        original_items = self.get_factor_items(factor_name)

        if len(original_items) == 0:
            logger.error(f"요인 '{factor_name}'의 문항을 찾을 수 없습니다.")
            return {'error': f"요인 '{factor_name}'의 문항을 찾을 수 없습니다."}

        logger.info(f"원본 문항 수: {len(original_items)}")

        # 현재 신뢰도 계산
        current_alpha = self.calculate_cronbach_alpha(original_items)
        current_cr, current_ave = self.calculate_cr_and_ave(factor_name, original_items)

        logger.info(f"현재 신뢰도 - Alpha: {current_alpha:.4f}, CR: {current_cr:.4f}, AVE: {current_ave:.4f}")

        # 최적화 결과 저장
        optimization_results = {
            'factor_name': factor_name,
            'original_items': original_items,
            'original_stats': {
                'cronbach_alpha': current_alpha,
                'composite_reliability': current_cr,
                'ave': current_ave,
                'n_items': len(original_items)
            },
            'optimization_attempts': [],
            'best_solution': None
        }

        # 문항 제거 조합 시도
        best_solution = None
        best_score = -1

        # 1개부터 max_removals개까지 문항 제거 시도
        for n_remove in range(1, min(max_removals + 1, len(original_items) - self.RELIABILITY_THRESHOLDS['min_items'] + 1)):
            logger.info(f"{n_remove}개 문항 제거 조합 시도 중...")

            # 제거할 문항들의 모든 조합 생성
            for items_to_remove in combinations(original_items, n_remove):
                remaining_items = [item for item in original_items if item not in items_to_remove]

                # 최소 문항 수 확인
                if len(remaining_items) < self.RELIABILITY_THRESHOLDS['min_items']:
                    continue

                # 신뢰도 계산
                alpha = self.calculate_cronbach_alpha(remaining_items)
                cr, ave = self.calculate_cr_and_ave(factor_name, remaining_items)

                # 기준 충족 여부 확인
                meets_criteria = (
                    alpha >= self.RELIABILITY_THRESHOLDS['cronbach_alpha'] and
                    cr >= self.RELIABILITY_THRESHOLDS['composite_reliability'] and
                    ave >= self.RELIABILITY_THRESHOLDS['ave']
                )

                # 점수 계산 (모든 기준을 만족하면서 문항 수가 많을수록 좋음)
                if meets_criteria:
                    score = len(remaining_items) + (alpha + cr + ave) / 3

                    attempt_result = {
                        'items_removed': list(items_to_remove),
                        'remaining_items': remaining_items,
                        'n_remaining': len(remaining_items),
                        'cronbach_alpha': alpha,
                        'composite_reliability': cr,
                        'ave': ave,
                        'meets_all_criteria': meets_criteria,
                        'score': score
                    }

                    optimization_results['optimization_attempts'].append(attempt_result)

                    # 최고 점수 업데이트
                    if score > best_score:
                        best_score = score
                        best_solution = attempt_result.copy()
                        logger.info(f"새로운 최적해 발견 - 점수: {score:.4f}, 남은 문항: {len(remaining_items)}개")

            # 해결책을 찾았으면 더 많은 문항 제거는 시도하지 않음
            if best_solution is not None:
                break

        optimization_results['best_solution'] = best_solution

        if best_solution:
            logger.info(f"최적화 완료 - 제거할 문항: {len(best_solution['items_removed'])}개")
            logger.info(f"최종 신뢰도 - Alpha: {best_solution['cronbach_alpha']:.4f}, "
                       f"CR: {best_solution['composite_reliability']:.4f}, "
                       f"AVE: {best_solution['ave']:.4f}")
        else:
            logger.warning(f"요인 '{factor_name}'에 대한 최적해를 찾지 못했습니다.")

        return optimization_results

    def optimize_all_problematic_factors(self, max_removals: int = 10) -> Dict[str, Any]:
        """
        모든 문제 요인들의 신뢰도 최적화

        Args:
            max_removals (int): 각 요인별 최대 제거할 문항 수

        Returns:
            Dict[str, Any]: 전체 최적화 결과
        """
        logger.info("모든 문제 요인 신뢰도 최적화 시작")

        problematic_factors = self.identify_problematic_factors()

        if not problematic_factors:
            logger.info("최적화가 필요한 요인이 없습니다.")
            return {'message': '최적화가 필요한 요인이 없습니다.'}

        all_results = {
            'problematic_factors': problematic_factors,
            'optimization_results': {},
            'summary': {
                'total_factors': len(problematic_factors),
                'successfully_optimized': 0,
                'failed_optimization': 0
            }
        }

        for factor_name in problematic_factors:
            logger.info(f"요인 '{factor_name}' 최적화 중...")

            result = self.optimize_factor_reliability(factor_name, max_removals)
            all_results['optimization_results'][factor_name] = result

            if result.get('best_solution'):
                all_results['summary']['successfully_optimized'] += 1
            else:
                all_results['summary']['failed_optimization'] += 1

        logger.info(f"전체 최적화 완료 - 성공: {all_results['summary']['successfully_optimized']}개, "
                   f"실패: {all_results['summary']['failed_optimization']}개")

        return all_results

    def generate_optimization_report(self, optimization_results: Dict[str, Any],
                                   output_dir: str = "reliability_optimization_results") -> bool:
        """
        최적화 결과 보고서 생성

        Args:
            optimization_results (Dict[str, Any]): 최적화 결과
            output_dir (str): 출력 디렉토리

        Returns:
            bool: 보고서 생성 성공 여부
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            # 1. 요약 보고서 생성
            summary_data = []
            detailed_data = []

            for factor_name, result in optimization_results['optimization_results'].items():
                if 'error' in result:
                    continue

                original_stats = result['original_stats']
                best_solution = result['best_solution']

                summary_row = {
                    'Factor': factor_name,
                    'Original_Items': original_stats['n_items'],
                    'Original_Alpha': original_stats['cronbach_alpha'],
                    'Original_CR': original_stats['composite_reliability'],
                    'Original_AVE': original_stats['ave'],
                    'Optimization_Success': best_solution is not None
                }

                if best_solution:
                    summary_row.update({
                        'Optimized_Items': best_solution['n_remaining'],
                        'Items_Removed': len(best_solution['items_removed']),
                        'Optimized_Alpha': best_solution['cronbach_alpha'],
                        'Optimized_CR': best_solution['composite_reliability'],
                        'Optimized_AVE': best_solution['ave'],
                        'Meets_All_Criteria': best_solution['meets_all_criteria'],
                        'Removed_Items': ', '.join(best_solution['items_removed'])
                    })
                else:
                    summary_row.update({
                        'Optimized_Items': 'N/A',
                        'Items_Removed': 'N/A',
                        'Optimized_Alpha': 'N/A',
                        'Optimized_CR': 'N/A',
                        'Optimized_AVE': 'N/A',
                        'Meets_All_Criteria': False,
                        'Removed_Items': 'N/A'
                    })

                summary_data.append(summary_row)

                # 상세 결과 데이터
                for attempt in result['optimization_attempts']:
                    detailed_row = {
                        'Factor': factor_name,
                        'Items_Removed': ', '.join(attempt['items_removed']),
                        'Remaining_Items': len(attempt['remaining_items']),
                        'Cronbach_Alpha': attempt['cronbach_alpha'],
                        'Composite_Reliability': attempt['composite_reliability'],
                        'AVE': attempt['ave'],
                        'Meets_All_Criteria': attempt['meets_all_criteria'],
                        'Score': attempt['score']
                    }
                    detailed_data.append(detailed_row)

            # CSV 파일 저장
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(output_path / "optimization_summary.csv", index=False, encoding='utf-8-sig')
                logger.info(f"요약 보고서 저장: {output_path / 'optimization_summary.csv'}")

            if detailed_data:
                detailed_df = pd.DataFrame(detailed_data)
                detailed_df.to_csv(output_path / "optimization_detailed.csv", index=False, encoding='utf-8-sig')
                logger.info(f"상세 보고서 저장: {output_path / 'optimization_detailed.csv'}")

            # JSON 형태로도 저장
            with open(output_path / "optimization_results.json", 'w', encoding='utf-8') as f:
                json.dump(optimization_results, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"최적화 결과 보고서 생성 완료: {output_path}")
            return True

        except Exception as e:
            logger.error(f"보고서 생성 중 오류: {e}")
            return False

    def print_optimization_summary(self, optimization_results: Dict[str, Any]) -> None:
        """
        최적화 결과 요약 출력

        Args:
            optimization_results (Dict[str, Any]): 최적화 결과
        """
        print("\n" + "="*80)
        print("🔧 신뢰도 최적화 결과 요약")
        print("="*80)

        if 'message' in optimization_results:
            print(f"📋 {optimization_results['message']}")
            return

        summary = optimization_results['summary']
        print(f"📊 분석 대상 요인: {summary['total_factors']}개")
        print(f"✅ 최적화 성공: {summary['successfully_optimized']}개")
        print(f"❌ 최적화 실패: {summary['failed_optimization']}개")

        print("\n" + "-"*80)
        print("📋 요인별 최적화 결과")
        print("-"*80)

        for factor_name, result in optimization_results['optimization_results'].items():
            if 'error' in result:
                print(f"\n❌ {factor_name}: {result['error']}")
                continue

            original_stats = result['original_stats']
            best_solution = result['best_solution']

            print(f"\n🔹 {factor_name}")
            print(f"   📈 원본 신뢰도:")
            print(f"      - 문항 수: {original_stats['n_items']}개")
            print(f"      - Cronbach's α: {original_stats['cronbach_alpha']:.4f}")
            print(f"      - CR: {original_stats['composite_reliability']:.4f}")
            print(f"      - AVE: {original_stats['ave']:.4f}")

            if best_solution:
                print(f"   ✨ 최적화 결과:")
                print(f"      - 제거 문항: {len(best_solution['items_removed'])}개 ({', '.join(best_solution['items_removed'])})")
                print(f"      - 남은 문항: {best_solution['n_remaining']}개")
                print(f"      - Cronbach's α: {best_solution['cronbach_alpha']:.4f}")
                print(f"      - CR: {best_solution['composite_reliability']:.4f}")
                print(f"      - AVE: {best_solution['ave']:.4f}")
                print(f"      - 모든 기준 충족: {'✅' if best_solution['meets_all_criteria'] else '❌'}")
            else:
                print(f"   ❌ 최적화 실패: 기준을 만족하는 해결책을 찾지 못했습니다.")

        print("\n" + "="*80)
        print("🎯 최적화 완료!")
        print("="*80)
