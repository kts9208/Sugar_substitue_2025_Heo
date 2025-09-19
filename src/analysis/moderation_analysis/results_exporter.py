"""
Moderation Analysis Results Exporter Module

조절효과 분석 결과를 CSV, JSON, 요약보고서 형태로 저장하는 모듈입니다.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from datetime import datetime
import logging

from .config import ModerationAnalysisConfig

logger = logging.getLogger(__name__)


class ModerationResultsExporter:
    """조절효과 분석 결과 저장 클래스"""
    
    def __init__(self, config: Optional[ModerationAnalysisConfig] = None):
        """
        결과 저장기 초기화
        
        Args:
            config (Optional[ModerationAnalysisConfig]): 분석 설정
        """
        from .config import DEFAULT_CONFIG
        self.config = config or DEFAULT_CONFIG
        self.results_dir = Path(self.config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 타임스탬프 생성
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"결과 저장기 초기화: {self.results_dir}")
    
    def export_comprehensive_results(self, results: Dict[str, Any],
                                   analysis_name: Optional[str] = None) -> Dict[str, Path]:
        """
        포괄적 결과 저장 (CSV, JSON, 보고서)
        
        Args:
            results (Dict[str, Any]): 분석 결과
            analysis_name (Optional[str]): 분석명 (파일명에 사용)
            
        Returns:
            Dict[str, Path]: 저장된 파일 경로들
        """
        logger.info("포괄적 결과 저장 시작")
        
        # 분석명 설정
        if analysis_name is None:
            vars_info = results.get('variables', {})
            analysis_name = f"{vars_info.get('independent', 'X')}_x_{vars_info.get('moderator', 'Z')}_to_{vars_info.get('dependent', 'Y')}"
        
        saved_files = {}
        
        try:
            # 1. CSV 파일들 저장
            if self.config.save_csv:
                csv_files = self._save_csv_results(results, analysis_name)
                saved_files.update(csv_files)
            
            # 2. JSON 파일 저장
            if self.config.save_json:
                json_file = self._save_json_results(results, analysis_name)
                saved_files['json'] = json_file
            
            # 3. 요약 보고서 저장
            if self.config.save_report:
                report_file = self._save_summary_report(results, analysis_name)
                saved_files['report'] = report_file
            
            logger.info(f"결과 저장 완료: {len(saved_files)}개 파일")
            return saved_files
            
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
            raise
    
    def _save_csv_results(self, results: Dict[str, Any], analysis_name: str) -> Dict[str, Path]:
        """CSV 결과 파일들 저장"""
        csv_files = {}
        
        # 1. 회귀계수 테이블
        coefficients_df = self._create_coefficients_table(results.get('coefficients', {}))
        if not coefficients_df.empty:
            coeff_file = self.results_dir / f"{analysis_name}_coefficients_{self.timestamp}.csv"
            coefficients_df.to_csv(coeff_file, index=True, encoding='utf-8-sig')
            csv_files['coefficients'] = coeff_file
        
        # 2. 단순기울기 분석 결과
        simple_slopes_df = self._create_simple_slopes_table(results.get('simple_slopes', {}))
        if not simple_slopes_df.empty:
            slopes_file = self.results_dir / f"{analysis_name}_simple_slopes_{self.timestamp}.csv"
            simple_slopes_df.to_csv(slopes_file, index=True, encoding='utf-8-sig')
            csv_files['simple_slopes'] = slopes_file
        
        # 3. 조건부 효과 결과
        conditional_df = self._create_conditional_effects_table(results.get('conditional_effects', {}))
        if not conditional_df.empty:
            conditional_file = self.results_dir / f"{analysis_name}_conditional_effects_{self.timestamp}.csv"
            conditional_df.to_csv(conditional_file, index=True, encoding='utf-8-sig')
            csv_files['conditional_effects'] = conditional_file
        
        # 4. 적합도 지수
        fit_indices_df = self._create_fit_indices_table(results.get('fit_indices', {}))
        if not fit_indices_df.empty:
            fit_file = self.results_dir / f"{analysis_name}_fit_indices_{self.timestamp}.csv"
            fit_indices_df.to_csv(fit_file, index=True, encoding='utf-8-sig')
            csv_files['fit_indices'] = fit_file
        
        return csv_files
    
    def _create_coefficients_table(self, coefficients: Dict[str, Any]) -> pd.DataFrame:
        """회귀계수 테이블 생성"""
        if not coefficients:
            return pd.DataFrame()
        
        coeff_data = []
        for var_name, coeff_info in coefficients.items():
            coeff_data.append({
                'Variable': var_name,
                'Estimate': coeff_info.get('estimate', np.nan),
                'Std_Error': coeff_info.get('std_error', np.nan),
                'Z_Value': coeff_info.get('z_value', np.nan),
                'P_Value': coeff_info.get('p_value', np.nan),
                'Std_Estimate': coeff_info.get('std_estimate', np.nan),
                'Significant': coeff_info.get('significant', False)
            })
        
        return pd.DataFrame(coeff_data).set_index('Variable')
    
    def _create_simple_slopes_table(self, simple_slopes: Dict[str, Any]) -> pd.DataFrame:
        """단순기울기 테이블 생성"""
        if not simple_slopes:
            return pd.DataFrame()
        
        slopes_data = []
        for level, slope_info in simple_slopes.items():
            slopes_data.append({
                'Moderator_Level': level,
                'Moderator_Value': slope_info.get('moderator_value', np.nan),
                'Simple_Slope': slope_info.get('simple_slope', np.nan),
                'Std_Error': slope_info.get('std_error', np.nan),
                'T_Value': slope_info.get('t_value', np.nan),
                'P_Value': slope_info.get('p_value', np.nan),
                'Significant': slope_info.get('significant', False)
            })
        
        return pd.DataFrame(slopes_data).set_index('Moderator_Level')
    
    def _create_conditional_effects_table(self, conditional_effects: Dict[str, Any]) -> pd.DataFrame:
        """조건부 효과 테이블 생성"""
        if not conditional_effects:
            return pd.DataFrame()
        
        conditional_data = []
        for percentile, effect_info in conditional_effects.items():
            conditional_data.append({
                'Percentile': percentile,
                'Moderator_Value': effect_info.get('moderator_value', np.nan),
                'Conditional_Effect': effect_info.get('simple_slope', np.nan),
                'Std_Error': effect_info.get('std_error', np.nan),
                'T_Value': effect_info.get('t_value', np.nan),
                'P_Value': effect_info.get('p_value', np.nan),
                'Significant': effect_info.get('significant', False)
            })
        
        return pd.DataFrame(conditional_data).set_index('Percentile')
    
    def _create_fit_indices_table(self, fit_indices: Dict[str, float]) -> pd.DataFrame:
        """적합도 지수 테이블 생성"""
        if not fit_indices:
            return pd.DataFrame()
        
        fit_data = []
        for index_name, value in fit_indices.items():
            # 적합도 해석
            interpretation = self._interpret_fit_index(index_name, value)
            
            fit_data.append({
                'Fit_Index': index_name,
                'Value': value,
                'Interpretation': interpretation
            })
        
        return pd.DataFrame(fit_data).set_index('Fit_Index')
    
    def _interpret_fit_index(self, index_name: str, value: float) -> str:
        """적합도 지수 해석"""
        if index_name in ['CFI', 'TLI']:
            if value >= 0.95:
                return 'Excellent'
            elif value >= 0.90:
                return 'Good'
            else:
                return 'Poor'
        elif index_name == 'RMSEA':
            if value <= 0.05:
                return 'Excellent'
            elif value <= 0.08:
                return 'Good'
            else:
                return 'Poor'
        elif index_name == 'SRMR':
            if value <= 0.05:
                return 'Excellent'
            elif value <= 0.08:
                return 'Good'
            else:
                return 'Poor'
        else:
            return 'N/A'
    
    def _save_json_results(self, results: Dict[str, Any], analysis_name: str) -> Path:
        """JSON 결과 파일 저장"""
        json_file = self.results_dir / f"{analysis_name}_full_results_{self.timestamp}.json"
        
        # JSON 직렬화 가능하도록 변환
        json_results = self._convert_to_json_serializable(results)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        return json_file
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """JSON 직렬화 가능한 형태로 변환"""
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def _save_summary_report(self, results: Dict[str, Any], analysis_name: str) -> Path:
        """요약 보고서 저장"""
        report_file = self.results_dir / f"{analysis_name}_summary_report_{self.timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_summary_report(results, analysis_name))
        
        return report_file
    
    def _generate_summary_report(self, results: Dict[str, Any], analysis_name: str) -> str:
        """요약 보고서 생성"""
        report_lines = []
        
        # 헤더
        report_lines.append("=" * 80)
        report_lines.append("조절효과 분석 요약 보고서")
        report_lines.append("=" * 80)
        report_lines.append(f"분석명: {analysis_name}")
        report_lines.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 변수 정보
        variables = results.get('variables', {})
        report_lines.append("📋 분석 변수")
        report_lines.append("-" * 40)
        report_lines.append(f"독립변수: {variables.get('independent', 'N/A')}")
        report_lines.append(f"종속변수: {variables.get('dependent', 'N/A')}")
        report_lines.append(f"조절변수: {variables.get('moderator', 'N/A')}")
        report_lines.append(f"상호작용항: {variables.get('interaction', 'N/A')}")
        report_lines.append("")
        
        # 모델 정보
        model_info = results.get('model_info', {})
        report_lines.append("📊 모델 정보")
        report_lines.append("-" * 40)
        report_lines.append(f"관측치 수: {model_info.get('n_observations', 'N/A')}")
        report_lines.append(f"모수 수: {model_info.get('n_parameters', 'N/A')}")
        report_lines.append("")
        
        # 조절효과 검정 결과
        moderation_test = results.get('moderation_test', {})
        report_lines.append("🎯 조절효과 검정 결과")
        report_lines.append("-" * 40)
        interaction_coeff = moderation_test.get('interaction_coefficient', 'N/A')
        std_error = moderation_test.get('std_error', 'N/A')
        z_value = moderation_test.get('z_value', 'N/A')
        p_value = moderation_test.get('p_value', 'N/A')

        report_lines.append(f"상호작용 계수: {interaction_coeff:.4f}" if isinstance(interaction_coeff, (int, float)) else f"상호작용 계수: {interaction_coeff}")
        report_lines.append(f"표준오차: {std_error:.4f}" if isinstance(std_error, (int, float)) else f"표준오차: {std_error}")
        report_lines.append(f"Z값: {z_value:.4f}" if isinstance(z_value, (int, float)) else f"Z값: {z_value}")
        report_lines.append(f"P값: {p_value:.4f}" if isinstance(p_value, (int, float)) else f"P값: {p_value}")
        report_lines.append(f"유의성: {'유의함' if moderation_test.get('significant', False) else '유의하지 않음'}")
        report_lines.append(f"해석: {moderation_test.get('interpretation', 'N/A')}")
        report_lines.append("")
        
        # 단순기울기 분석
        simple_slopes = results.get('simple_slopes', {})
        if simple_slopes:
            report_lines.append("📈 단순기울기 분석")
            report_lines.append("-" * 40)
            for level, slope_info in simple_slopes.items():
                report_lines.append(f"{level.upper()}:")

                moderator_value = slope_info.get('moderator_value', 'N/A')
                simple_slope = slope_info.get('simple_slope', 'N/A')
                p_value = slope_info.get('p_value', 'N/A')

                report_lines.append(f"  조절변수 값: {moderator_value:.4f}" if isinstance(moderator_value, (int, float)) else f"  조절변수 값: {moderator_value}")
                report_lines.append(f"  단순기울기: {simple_slope:.4f}" if isinstance(simple_slope, (int, float)) else f"  단순기울기: {simple_slope}")
                report_lines.append(f"  P값: {p_value:.4f}" if isinstance(p_value, (int, float)) else f"  P값: {p_value}")
                report_lines.append(f"  유의성: {'유의함' if slope_info.get('significant', False) else '유의하지 않음'}")
                report_lines.append("")
        
        # 적합도 지수
        fit_indices = results.get('fit_indices', {})
        if fit_indices:
            report_lines.append("📏 모델 적합도")
            report_lines.append("-" * 40)
            for index_name, value in fit_indices.items():
                interpretation = self._interpret_fit_index(index_name, value)
                if isinstance(value, (int, float)):
                    report_lines.append(f"{index_name}: {value:.4f} ({interpretation})")
                else:
                    report_lines.append(f"{index_name}: {value} ({interpretation})")
            report_lines.append("")
        
        # 결론
        report_lines.append("💡 분석 결론")
        report_lines.append("-" * 40)
        if moderation_test.get('significant', False):
            report_lines.append("✅ 조절효과가 통계적으로 유의합니다.")
            report_lines.append(f"   {moderation_test.get('interpretation', '')}")
        else:
            report_lines.append("❌ 조절효과가 통계적으로 유의하지 않습니다.")
        
        return "\n".join(report_lines)

    def save_comprehensive_results(self, comprehensive_results: Dict[str, Any],
                                 analysis_name: str = "comprehensive_analysis") -> Dict[str, Path]:
        """종합 조절효과 분석 결과 저장"""
        return save_comprehensive_moderation_results(comprehensive_results, analysis_name, self.config)


# 편의 함수들
def export_moderation_results(results: Dict[str, Any], analysis_name: Optional[str] = None,
                            config: Optional[ModerationAnalysisConfig] = None) -> Dict[str, Path]:
    """조절효과 분석 결과 저장 편의 함수"""
    exporter = ModerationResultsExporter(config)
    return exporter.export_comprehensive_results(results, analysis_name)


def create_moderation_report(results: Dict[str, Any], analysis_name: str,
                           config: Optional[ModerationAnalysisConfig] = None) -> Path:
    """조절효과 분석 보고서 생성 편의 함수"""
    exporter = ModerationResultsExporter(config)
    return exporter._save_summary_report(results, analysis_name)


def save_comprehensive_moderation_results(comprehensive_results: Dict[str, Any],
                                        analysis_name: str = "comprehensive_analysis",
                                        config: Optional[ModerationAnalysisConfig] = None) -> Dict[str, Path]:
    """종합 조절효과 분석 결과 저장 편의 함수"""
    exporter = ModerationResultsExporter(config)

    saved_files = {}

    try:
        # CSV 저장
        if 'detailed_results' in comprehensive_results:
            df = pd.DataFrame(comprehensive_results['detailed_results'])
            csv_path = exporter.results_dir / f"{analysis_name}_{exporter.timestamp}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            saved_files['csv_file'] = csv_path

        # JSON 저장 (JSON 직렬화 가능한 형태로 변환)
        json_data = {}
        for key, value in comprehensive_results.items():
            if key == 'detailed_results':
                # DataFrame을 dict로 변환
                json_data[key] = pd.DataFrame(value).to_dict('records') if value else []
            else:
                json_data[key] = value

        json_path = exporter.results_dir / f"{analysis_name}_{exporter.timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
        saved_files['json_file'] = json_path

        # 요약 보고서 저장
        report_path = exporter.results_dir / f"{analysis_name}_summary_{exporter.timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("5개 요인 간 조절효과 분석 종합 보고서\n")
            f.write("=" * 80 + "\n")
            f.write(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 요약 정보
            if 'summary' in comprehensive_results:
                summary = comprehensive_results['summary']
                f.write("📊 분석 요약\n")
                f.write("-" * 40 + "\n")
                f.write(f"총 분석 조합: {summary.get('total_combinations', 0)}개\n")
                f.write(f"성공한 분석: {summary.get('successful_analyses', 0)}개 ({summary.get('success_rate', 0):.1f}%)\n")
                f.write(f"유의한 조절효과: {summary.get('significant_effects', 0)}개 ({summary.get('significance_rate', 0):.1f}%)\n\n")

                if summary.get('significant_effects', 0) == 0:
                    f.write("💡 유의한 조절효과가 발견되지 않았습니다.\n\n")

            # 요인별 분석 결과
            if 'variables' in comprehensive_results and 'detailed_results' in comprehensive_results:
                variables = comprehensive_results['variables']
                detailed_results = comprehensive_results['detailed_results']

                f.write("📋 요인별 분석 결과\n")
                f.write("-" * 40 + "\n\n")

                for var in variables:
                    # 해당 변수가 종속변수인 경우들 찾기
                    var_results = [r for r in detailed_results if r.get('dependent') == var]
                    significant_count = len([r for r in var_results if r.get('significant', False)])

                    f.write(f"{var} (종속변수):\n")
                    f.write(f"  총 분석: {len(var_results)}개\n")
                    f.write(f"  유의한 조절효과: {significant_count}개\n\n")

        saved_files['report_file'] = report_path

        logger.info(f"종합 결과 저장 완료: {len(saved_files)}개 파일")
        return saved_files

    except Exception as e:
        logger.error(f"종합 결과 저장 실패: {e}")
        return saved_files
