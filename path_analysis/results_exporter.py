"""
Path Analysis Results Exporter

경로분석 결과를 다양한 형태로 저장하는 모듈입니다.
CSV, Excel, JSON 형태로 결과를 내보낼 수 있습니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path
import json
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


class PathResultsExporter:
    """경로분석 결과 내보내기 클래스"""
    
    def __init__(self, output_dir: str = "path_analysis_results"):
        """
        초기화
        
        Args:
            output_dir (str): 결과 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PathResultsExporter 초기화 완료: {self.output_dir}")
    
    def export_comprehensive_results(self, 
                                   analysis_results: Dict[str, Any],
                                   filename_prefix: str = "path_analysis") -> Dict[str, str]:
        """
        종합적인 결과 내보내기
        
        Args:
            analysis_results (Dict[str, Any]): 분석 결과
            filename_prefix (str): 파일명 접두사
            
        Returns:
            Dict[str, str]: 저장된 파일들의 경로
        """
        logger.info("종합적인 경로분석 결과 내보내기 시작")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        try:
            # 1. 모델 정보 저장
            if 'model_info' in analysis_results:
                model_info_file = self._export_model_info(
                    analysis_results['model_info'], 
                    f"{filename_prefix}_model_info_{timestamp}"
                )
                saved_files['model_info'] = str(model_info_file)
            
            # 2. 적합도 지수 저장
            if 'fit_indices' in analysis_results:
                fit_indices_file = self._export_fit_indices(
                    analysis_results['fit_indices'],
                    f"{filename_prefix}_fit_indices_{timestamp}"
                )
                saved_files['fit_indices'] = str(fit_indices_file)
            
            # 3. 구조적 경로계수 저장 (잠재변수간만)
            if 'path_coefficients' in analysis_results:
                path_coeffs_file = self._export_path_coefficients(
                    analysis_results['path_coefficients'],
                    f"{filename_prefix}_structural_paths_{timestamp}"
                )
                saved_files['structural_paths'] = str(path_coeffs_file)

            # 3.5. 경로 분석 결과 저장
            if 'path_analysis' in analysis_results:
                path_analysis_file = self._export_path_analysis(
                    analysis_results['path_analysis'],
                    f"{filename_prefix}_path_analysis_{timestamp}"
                )
                saved_files['path_analysis'] = str(path_analysis_file)
            
            # 4. 효과 분석 결과 저장
            if 'effects_analysis' in analysis_results:
                effects_file = self._export_effects_analysis(
                    analysis_results['effects_analysis'],
                    f"{filename_prefix}_effects_{timestamp}"
                )
                saved_files['effects_analysis'] = str(effects_file)
            
            # 5. 부트스트래핑 결과 저장
            if 'bootstrap_effects' in analysis_results:
                bootstrap_file = self._export_bootstrap_results(
                    analysis_results['bootstrap_effects'],
                    f"{filename_prefix}_bootstrap_{timestamp}"
                )
                saved_files['bootstrap_results'] = str(bootstrap_file)

            # 6. 모든 매개효과 분석 결과 저장
            if 'all_mediations' in analysis_results:
                mediations_file = self._export_all_mediations(
                    analysis_results['all_mediations'],
                    f"{filename_prefix}_all_mediations_{timestamp}"
                )
                saved_files['all_mediations'] = str(mediations_file)

            # 7. 전체 결과 JSON 저장
            json_file = self._export_full_results_json(
                analysis_results,
                f"{filename_prefix}_full_results_{timestamp}"
            )
            saved_files['full_results_json'] = str(json_file)

            # 8. 요약 보고서 생성
            summary_file = self._create_summary_report(
                analysis_results,
                f"{filename_prefix}_summary_{timestamp}"
            )
            saved_files['summary_report'] = str(summary_file)
            
            logger.info(f"결과 내보내기 완료: {len(saved_files)}개 파일 저장")
            return saved_files
            
        except Exception as e:
            logger.error(f"결과 내보내기 중 오류: {e}")
            raise
    
    def _export_model_info(self, model_info: Dict[str, Any], filename: str) -> Path:
        """모델 정보 저장"""
        try:
            df = pd.DataFrame([model_info])
            file_path = self.output_dir / f"{filename}.csv"
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"모델 정보 저장 완료: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"모델 정보 저장 오류: {e}")
            raise
    
    def _export_fit_indices(self, fit_indices: Dict[str, float], filename: str) -> Path:
        """적합도 지수 저장"""
        try:
            # 적합도 지수를 DataFrame으로 변환
            fit_data = []
            for index_name, value in fit_indices.items():
                # Series나 다른 타입을 숫자로 변환
                if hasattr(value, 'iloc'):
                    numeric_value = value.iloc[0] if len(value) > 0 else np.nan
                else:
                    numeric_value = value

                fit_data.append({
                    'Fit_Index': index_name,
                    'Value': numeric_value,
                    'Interpretation': self._interpret_fit_index(index_name, numeric_value)
                })
            
            df = pd.DataFrame(fit_data)
            file_path = self.output_dir / f"{filename}.csv"
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"적합도 지수 저장 완료: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"적합도 지수 저장 오류: {e}")
            raise
    
    def _interpret_fit_index(self, index_name: str, value: Any) -> str:
        """적합도 지수 해석"""
        # Series나 다른 타입 처리
        if hasattr(value, 'iloc'):
            value = value.iloc[0] if len(value) > 0 else np.nan

        if pd.isna(value):
            return "N/A"
        
        interpretations = {
            'chi_square': 'Lower is better (non-significant preferred)',
            'p_value': 'Non-significant (>0.05) preferred',
            'cfi': 'Good: >0.95, Acceptable: >0.90',
            'tli': 'Good: >0.95, Acceptable: >0.90',
            'rmsea': 'Good: <0.06, Acceptable: <0.08',
            'srmr': 'Good: <0.08, Acceptable: <0.10',
            'aic': 'Lower is better (for model comparison)',
            'bic': 'Lower is better (for model comparison)'
        }
        
        base_interpretation = interpretations.get(index_name, 'See literature for interpretation')
        
        # 구체적인 평가 추가
        if index_name == 'cfi' or index_name == 'tli':
            if value > 0.95:
                return f"{base_interpretation} - Excellent fit"
            elif value > 0.90:
                return f"{base_interpretation} - Acceptable fit"
            else:
                return f"{base_interpretation} - Poor fit"
        elif index_name == 'rmsea':
            if value < 0.06:
                return f"{base_interpretation} - Good fit"
            elif value < 0.08:
                return f"{base_interpretation} - Acceptable fit"
            else:
                return f"{base_interpretation} - Poor fit"
        elif index_name == 'p_value':
            if value > 0.05:
                return f"{base_interpretation} - Good fit (non-significant)"
            else:
                return f"{base_interpretation} - Poor fit (significant)"
        
        return base_interpretation
    
    def _export_path_coefficients(self, path_coefficients: Dict[str, Any], filename: str) -> Path:
        """구조적 경로계수 저장 (잠재변수간 경로만)"""
        try:
            # 경로계수 데이터 정리
            path_data = []

            if 'paths' in path_coefficients and 'coefficients' in path_coefficients:
                paths = path_coefficients['paths']
                coefficients = path_coefficients['coefficients']
                standard_errors = path_coefficients.get('standard_errors', {})
                z_values = path_coefficients.get('z_values', {})
                p_values = path_coefficients.get('p_values', {})

                for i, (from_var, to_var) in enumerate(paths):
                    path_key = f"{from_var} -> {to_var}"

                    path_info = {
                        'From_Variable': from_var,
                        'To_Variable': to_var,
                        'Path': path_key,
                        'Coefficient': coefficients.get(i, np.nan),
                        'Standard_Error': standard_errors.get(i, np.nan),
                        'Z_Value': z_values.get(i, np.nan),
                        'P_Value': p_values.get(i, np.nan),
                        'Significance': self._get_significance_stars(p_values.get(i, np.nan))
                    }

                    path_data.append(path_info)

            # 추가 정보 포함
            if path_data:
                df = pd.DataFrame(path_data)

                # 메타데이터 추가
                metadata_info = {
                    'From_Variable': 'METADATA',
                    'To_Variable': 'INFO',
                    'Path': f"Total Structural Paths: {len(path_data)}",
                    'Coefficient': path_coefficients.get('n_structural_paths', len(path_data)),
                    'Standard_Error': np.nan,
                    'Z_Value': np.nan,
                    'P_Value': np.nan,
                    'Significance': ''
                }

                # 잠재변수 목록 추가
                if 'latent_variables' in path_coefficients:
                    latent_vars = path_coefficients['latent_variables']
                    latent_info = {
                        'From_Variable': 'LATENT_VARS',
                        'To_Variable': 'LIST',
                        'Path': f"Variables: {', '.join(latent_vars)}",
                        'Coefficient': len(latent_vars),
                        'Standard_Error': np.nan,
                        'Z_Value': np.nan,
                        'P_Value': np.nan,
                        'Significance': ''
                    }
                    df = pd.concat([df, pd.DataFrame([metadata_info, latent_info])], ignore_index=True)
                else:
                    df = pd.concat([df, pd.DataFrame([metadata_info])], ignore_index=True)
            else:
                df = pd.DataFrame(columns=['From_Variable', 'To_Variable', 'Path', 'Coefficient',
                                         'Standard_Error', 'Z_Value', 'P_Value', 'Significance'])

            file_path = self.output_dir / f"{filename}.csv"
            df.to_csv(file_path, index=False, encoding='utf-8-sig')

            logger.info(f"구조적 경로계수 저장 완료: {file_path} ({len(path_data)}개 경로)")
            return file_path

        except Exception as e:
            logger.error(f"경로계수 저장 오류: {e}")
            raise

    def _export_path_analysis(self, path_analysis: Dict[str, Any], filename: str) -> Path:
        """경로 분석 결과 저장"""
        try:
            # 경로 분석 데이터 정리
            analysis_data = []

            # 기본 정보
            basic_info = [
                {'Category': 'Basic Info', 'Item': 'Number of Latent Variables', 'Value': path_analysis.get('n_latent_variables', 0)},
                {'Category': 'Basic Info', 'Item': 'Latent Variables', 'Value': ', '.join(path_analysis.get('latent_variables', []))},
                {'Category': 'Path Coverage', 'Item': 'Total Possible Paths', 'Value': path_analysis.get('n_possible_paths', 0)},
                {'Category': 'Path Coverage', 'Item': 'Current Paths in Model', 'Value': path_analysis.get('n_current_paths', 0)},
                {'Category': 'Path Coverage', 'Item': 'Missing Paths', 'Value': path_analysis.get('n_missing_paths', 0)},
                {'Category': 'Path Coverage', 'Item': 'Coverage Ratio', 'Value': f"{path_analysis.get('coverage_ratio', 0):.1%}"}
            ]
            analysis_data.extend(basic_info)

            # 현재 경로 목록
            current_paths = path_analysis.get('current_paths', [])
            for i, (from_var, to_var) in enumerate(current_paths):
                analysis_data.append({
                    'Category': 'Current Paths',
                    'Item': f'Path {i+1}',
                    'Value': f'{from_var} → {to_var}'
                })

            # 누락된 경로 목록
            missing_paths = path_analysis.get('missing_paths', [])
            for i, (from_var, to_var) in enumerate(missing_paths):
                analysis_data.append({
                    'Category': 'Missing Paths',
                    'Item': f'Missing Path {i+1}',
                    'Value': f'{from_var} → {to_var}'
                })

            df = pd.DataFrame(analysis_data)
            file_path = self.output_dir / f"{filename}.csv"
            df.to_csv(file_path, index=False, encoding='utf-8-sig')

            logger.info(f"경로 분석 결과 저장 완료: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"경로 분석 결과 저장 오류: {e}")
            raise

    def _export_effects_analysis(self, effects_analysis: Dict[str, Any], filename: str) -> Path:
        """효과 분석 결과 저장"""
        try:
            effects_data = []
            
            # 직접효과
            if 'direct_effects' in effects_analysis:
                direct = effects_analysis['direct_effects']
                effects_data.append({
                    'Effect_Type': 'Direct Effect',
                    'Path': f"{effects_analysis.get('variables', {}).get('independent', 'X')} -> {effects_analysis.get('variables', {}).get('dependent', 'Y')}",
                    'Coefficient': direct.get('coefficient', np.nan),
                    'Standard_Error': direct.get('standard_error', np.nan),
                    'P_Value': direct.get('p_value', np.nan),
                    'Significance': self._get_significance_stars(direct.get('p_value', np.nan))
                })
            
            # 간접효과
            if 'indirect_effects' in effects_analysis:
                indirect = effects_analysis['indirect_effects']
                
                # 총 간접효과
                effects_data.append({
                    'Effect_Type': 'Total Indirect Effect',
                    'Path': 'All mediation paths',
                    'Coefficient': indirect.get('total_indirect_effect', np.nan),
                    'Standard_Error': np.nan,
                    'P_Value': np.nan,
                    'Significance': ''
                })
                
                # 개별 간접효과
                for mediator, path_info in indirect.get('individual_paths', {}).items():
                    effects_data.append({
                        'Effect_Type': f'Indirect Effect via {mediator}',
                        'Path': f"X -> {mediator} -> Y",
                        'Coefficient': path_info.get('indirect_effect', np.nan),
                        'Standard_Error': np.nan,
                        'P_Value': np.nan,
                        'Significance': ''
                    })
            
            # 총효과
            if 'total_effects' in effects_analysis:
                total = effects_analysis['total_effects']
                effects_data.append({
                    'Effect_Type': 'Total Effect',
                    'Path': 'Direct + Indirect',
                    'Coefficient': total.get('total_effect', np.nan),
                    'Standard_Error': np.nan,
                    'P_Value': np.nan,
                    'Significance': ''
                })
            
            df = pd.DataFrame(effects_data)
            file_path = self.output_dir / f"{filename}.csv"
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"효과 분석 결과 저장 완료: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"효과 분석 결과 저장 오류: {e}")
            raise

    def _export_bootstrap_results(self, bootstrap_results: Dict[str, Any], filename: str) -> Path:
        """부트스트래핑 결과 저장"""
        try:
            all_bootstrap_data = []

            for combination_key, bootstrap_result in bootstrap_results.items():
                # 기본 정보
                base_info = {
                    'Combination': combination_key,
                    'Category': 'Settings',
                    'Effect_Type': 'Bootstrap_Settings',
                    'Value': '',
                    'Lower_CI': '',
                    'Upper_CI': '',
                    'Significant': ''
                }

                # 설정 정보 추가
                settings = bootstrap_result.get('settings', {})
                for setting_key, setting_value in settings.items():
                    setting_row = base_info.copy()
                    setting_row['Effect_Type'] = f'Setting_{setting_key}'
                    setting_row['Value'] = str(setting_value)
                    all_bootstrap_data.append(setting_row)

                # 신뢰구간 결과 추가
                confidence_intervals = bootstrap_result.get('confidence_intervals', {})
                for effect_type, ci_info in confidence_intervals.items():
                    ci_row = base_info.copy()
                    ci_row['Category'] = 'Confidence_Intervals'
                    ci_row['Effect_Type'] = effect_type
                    ci_row['Value'] = ci_info.get('mean', '')
                    ci_row['Lower_CI'] = ci_info.get('lower_ci', '')
                    ci_row['Upper_CI'] = ci_info.get('upper_ci', '')
                    ci_row['Significant'] = 'Yes' if ci_info.get('significant', False) else 'No'
                    all_bootstrap_data.append(ci_row)

                # 부트스트래핑 통계 추가
                bootstrap_stats = bootstrap_result.get('bootstrap_statistics', {})
                for effect_type, stats_info in bootstrap_stats.items():
                    for stat_name, stat_value in stats_info.items():
                        stat_row = base_info.copy()
                        stat_row['Category'] = 'Bootstrap_Statistics'
                        stat_row['Effect_Type'] = f'{effect_type}_{stat_name}'
                        stat_row['Value'] = stat_value
                        all_bootstrap_data.append(stat_row)

            df = pd.DataFrame(all_bootstrap_data)
            file_path = self.output_dir / f"{filename}.csv"
            df.to_csv(file_path, index=False, encoding='utf-8-sig')

            logger.info(f"부트스트래핑 결과 저장 완료: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"부트스트래핑 결과 저장 오류: {e}")
            raise

    def _export_all_mediations(self, all_mediations: Dict[str, Any], filename: str) -> Path:
        """모든 매개효과 분석 결과 저장"""
        try:
            mediation_data = []

            # 요약 정보 먼저 추가
            summary = all_mediations.get('summary', {})
            summary_row = {
                'Analysis_Type': 'Summary',
                'Independent_Var': '',
                'Dependent_Var': '',
                'Mediator': '',
                'Indirect_Effect_Mean': '',
                'Lower_CI': '',
                'Upper_CI': '',
                'Significant': '',
                'Details': f"Total: {summary.get('total_combinations_tested', 0)}, "
                          f"Significant: {summary.get('significant_mediations_count', 0)}, "
                          f"Rate: {summary.get('significance_rate', 0):.1%}"
            }
            mediation_data.append(summary_row)

            # 유의한 매개효과 결과 추가
            significant_results = all_mediations.get('significant_results', {})
            for combination_key, result in significant_results.items():
                mediation_row = {
                    'Analysis_Type': 'Significant_Mediation',
                    'Independent_Var': result.get('independent_var', ''),
                    'Dependent_Var': result.get('dependent_var', ''),
                    'Mediator': result.get('mediator', ''),
                    'Indirect_Effect_Mean': result.get('indirect_effect_mean', ''),
                    'Lower_CI': result.get('indirect_effect_ci', [None, None])[0],
                    'Upper_CI': result.get('indirect_effect_ci', [None, None])[1],
                    'Significant': 'Yes' if result.get('is_significant', False) else 'No',
                    'Details': combination_key
                }
                mediation_data.append(mediation_row)

            # 모든 결과 추가 (유의하지 않은 것 포함)
            all_results = all_mediations.get('all_results', {})
            for combination_key, result in all_results.items():
                if combination_key not in significant_results:  # 이미 추가된 것은 제외
                    mediation_row = {
                        'Analysis_Type': 'All_Mediation',
                        'Independent_Var': result.get('independent_var', ''),
                        'Dependent_Var': result.get('dependent_var', ''),
                        'Mediator': result.get('mediator', ''),
                        'Indirect_Effect_Mean': result.get('indirect_effect_mean', ''),
                        'Lower_CI': result.get('indirect_effect_ci', [None, None])[0] if 'indirect_effect_ci' in result else '',
                        'Upper_CI': result.get('indirect_effect_ci', [None, None])[1] if 'indirect_effect_ci' in result else '',
                        'Significant': 'Yes' if result.get('is_significant', False) else 'No',
                        'Details': result.get('error', combination_key)
                    }
                    mediation_data.append(mediation_row)

            df = pd.DataFrame(mediation_data)
            file_path = self.output_dir / f"{filename}.csv"
            df.to_csv(file_path, index=False, encoding='utf-8-sig')

            logger.info(f"모든 매개효과 분석 결과 저장 완료: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"매개효과 분석 결과 저장 오류: {e}")
            raise
    
    def _export_full_results_json(self, analysis_results: Dict[str, Any], filename: str) -> Path:
        """전체 결과 JSON 저장"""
        try:
            # JSON 직렬화를 위한 데이터 정리
            json_data = self._prepare_for_json(analysis_results.copy())
            
            file_path = self.output_dir / f"{filename}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"전체 결과 JSON 저장 완료: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"JSON 저장 오류: {e}")
            raise
    
    def _prepare_for_json(self, data: Any) -> Any:
        """JSON 직렬화를 위한 데이터 준비"""
        if isinstance(data, dict):
            # semopy 모델 객체 제거
            if 'model_object' in data:
                del data['model_object']
            
            return {k: self._prepare_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, pd.Series):
            # Series는 먼저 처리
            return data.to_dict()
        elif isinstance(data, pd.DataFrame):
            return data.to_dict('records')
        elif isinstance(data, (int, float, str, bool)):
            return data
        elif data is None:
            return None
        else:
            # pandas의 isna 체크는 스칼라 값에만 적용
            try:
                if pd.isna(data):
                    return None
            except (ValueError, TypeError):
                pass
            return str(data)  # 기타 객체는 문자열로 변환
    
    def _create_summary_report(self, analysis_results: Dict[str, Any], filename: str) -> Path:
        """요약 보고서 생성"""
        try:
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("PATH ANALYSIS SUMMARY REPORT")
            report_lines.append("=" * 60)
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # 모델 정보
            if 'model_info' in analysis_results:
                model_info = analysis_results['model_info']
                report_lines.append("MODEL INFORMATION")
                report_lines.append("-" * 30)
                report_lines.append(f"Sample Size: {model_info.get('n_observations', 'N/A')}")
                report_lines.append(f"Variables: {model_info.get('n_variables', 'N/A')}")
                report_lines.append(f"Estimator: {model_info.get('estimator', 'N/A')}")
                report_lines.append("")
            
            # 적합도 지수
            if 'fit_indices' in analysis_results:
                fit_indices = analysis_results['fit_indices']
                report_lines.append("MODEL FIT INDICES")
                report_lines.append("-" * 30)
                for index_name, value in fit_indices.items():
                    # Series나 다른 타입 처리
                    if hasattr(value, 'iloc'):
                        numeric_value = value.iloc[0] if len(value) > 0 else np.nan
                    else:
                        numeric_value = value

                    try:
                        if not pd.isna(numeric_value) and isinstance(numeric_value, (int, float)):
                            report_lines.append(f"{index_name.upper()}: {numeric_value:.4f}")
                    except (ValueError, TypeError):
                        if numeric_value is not None:
                            report_lines.append(f"{index_name.upper()}: {numeric_value}")
                report_lines.append("")
            
            # 경로계수 요약
            if 'path_coefficients' in analysis_results:
                report_lines.append("PATH COEFFICIENTS SUMMARY")
                report_lines.append("-" * 30)
                path_coeffs = analysis_results['path_coefficients']
                if 'paths' in path_coeffs and 'coefficients' in path_coeffs:
                    for i, (from_var, to_var) in enumerate(path_coeffs['paths']):
                        coeff = path_coeffs['coefficients'].get(i, np.nan)
                        p_val = path_coeffs.get('p_values', {}).get(i, np.nan)
                        sig = self._get_significance_stars(p_val)
                        report_lines.append(f"{from_var} -> {to_var}: {coeff:.4f}{sig}")
                report_lines.append("")
            
            # 효과 분석 요약
            if 'effects_analysis' in analysis_results:
                effects = analysis_results['effects_analysis']
                report_lines.append("EFFECTS ANALYSIS SUMMARY")
                report_lines.append("-" * 30)

                if 'direct_effects' in effects:
                    direct_coeff = effects['direct_effects'].get('coefficient', np.nan)
                    report_lines.append(f"Direct Effect: {direct_coeff:.4f}")

                if 'indirect_effects' in effects:
                    indirect_coeff = effects['indirect_effects'].get('total_indirect_effect', np.nan)
                    report_lines.append(f"Indirect Effect: {indirect_coeff:.4f}")

                if 'total_effects' in effects:
                    total_coeff = effects['total_effects'].get('total_effect', np.nan)
                    report_lines.append(f"Total Effect: {total_coeff:.4f}")

                report_lines.append("")

            # 부트스트래핑 결과 요약
            if 'bootstrap_effects' in analysis_results:
                bootstrap_effects = analysis_results['bootstrap_effects']
                report_lines.append("BOOTSTRAP ANALYSIS SUMMARY")
                report_lines.append("-" * 30)
                report_lines.append(f"Bootstrap Combinations Analyzed: {len(bootstrap_effects)}")

                significant_count = 0
                for combination, result in bootstrap_effects.items():
                    ci_results = result.get('confidence_intervals', {})
                    for effect_type, ci_info in ci_results.items():
                        if ci_info.get('significant', False):
                            significant_count += 1

                report_lines.append(f"Significant Bootstrap Effects: {significant_count}")
                report_lines.append("")

            # 모든 매개효과 분석 요약
            if 'all_mediations' in analysis_results:
                all_mediations = analysis_results['all_mediations']
                summary = all_mediations.get('summary', {})
                report_lines.append("ALL MEDIATIONS ANALYSIS SUMMARY")
                report_lines.append("-" * 30)
                report_lines.append(f"Total Combinations Tested: {summary.get('total_combinations_tested', 0)}")
                report_lines.append(f"Significant Mediations: {summary.get('significant_mediations_count', 0)}")
                report_lines.append(f"Significance Rate: {summary.get('significance_rate', 0):.1%}")

                # 유의한 매개효과 상위 5개 표시
                significant_results = all_mediations.get('significant_results', {})
                if significant_results:
                    report_lines.append("\nTop Significant Mediations:")
                    sorted_mediations = sorted(
                        significant_results.items(),
                        key=lambda x: abs(x[1].get('indirect_effect_mean', 0)),
                        reverse=True
                    )[:5]

                    for combination_key, result in sorted_mediations:
                        effect_mean = result.get('indirect_effect_mean', 0)
                        report_lines.append(f"  {combination_key}: {effect_mean:.4f}")

                report_lines.append("")
            
            report_lines.append("=" * 60)
            
            # 파일 저장
            file_path = self.output_dir / f"{filename}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"요약 보고서 생성 완료: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"요약 보고서 생성 오류: {e}")
            raise
    
    def _get_significance_stars(self, p_value: float) -> str:
        """유의도 별표 반환"""
        if pd.isna(p_value):
            return ""
        elif p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        elif p_value < 0.1:
            return "."
        else:
            return ""


# 편의 함수
def export_path_results(analysis_results: Dict[str, Any],
                       output_dir: str = "path_analysis_results",
                       filename_prefix: str = "path_analysis") -> Dict[str, str]:
    """
    경로분석 결과 내보내기 편의 함수
    
    Args:
        analysis_results (Dict[str, Any]): 분석 결과
        output_dir (str): 출력 디렉토리
        filename_prefix (str): 파일명 접두사
        
    Returns:
        Dict[str, str]: 저장된 파일들의 경로
    """
    exporter = PathResultsExporter(output_dir)
    return exporter.export_comprehensive_results(analysis_results, filename_prefix)
