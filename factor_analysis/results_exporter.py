"""
Factor Analysis Results Exporter Module

이 모듈은 factor loading 분석 결과를 CSV 파일로 저장하는 기능을 제공합니다.
다양한 형태의 결과 테이블을 생성하고 내보낼 수 있습니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class FactorResultsExporter:
    """Factor Analysis 결과를 내보내는 클래스"""
    
    def __init__(self, output_dir: Union[str, Path] = None):
        """
        Results Exporter 초기화
        
        Args:
            output_dir (Union[str, Path]): 결과 저장 디렉토리
        """
        if output_dir is None:
            # 기본 경로: factor_analysis_results
            self.output_dir = Path("factor_analysis_results")
        else:
            self.output_dir = Path(output_dir)
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_factor_loadings(self, results: Dict[str, Any], 
                              filename: Optional[str] = None) -> Path:
        """
        Factor loadings를 CSV 파일로 내보내기
        
        Args:
            results (Dict[str, Any]): 분석 결과
            filename (Optional[str]): 파일명 (기본값: 자동 생성)
            
        Returns:
            Path: 저장된 파일 경로
        """
        if 'factor_loadings' not in results or results['factor_loadings'].empty:
            raise ValueError("Factor loadings 데이터가 없습니다")
        
        # 파일명 생성
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_type = results.get('analysis_type', 'factor_analysis')
            filename = f"factor_loadings_{analysis_type}_{timestamp}.csv"
        
        file_path = self.output_dir / filename
        
        # Factor loadings 테이블 준비
        loadings_df = results['factor_loadings'].copy()
        
        # 추가 정보 컬럼
        loadings_df['Analysis_Type'] = results.get('analysis_type', 'unknown')
        loadings_df['Sample_Size'] = results.get('model_info', {}).get('n_observations', 'unknown')
        
        # CSV 저장
        loadings_df.to_csv(file_path, index=False, encoding='utf-8-sig')
        logger.info(f"Factor loadings 저장 완료: {file_path}")
        
        return file_path
    
    def export_fit_indices(self, results: Dict[str, Any], 
                          filename: Optional[str] = None) -> Path:
        """
        적합도 지수를 CSV 파일로 내보내기
        
        Args:
            results (Dict[str, Any]): 분석 결과
            filename (Optional[str]): 파일명
            
        Returns:
            Path: 저장된 파일 경로
        """
        if 'fit_indices' not in results or not results['fit_indices']:
            raise ValueError("적합도 지수 데이터가 없습니다")
        
        # 파일명 생성
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_type = results.get('analysis_type', 'factor_analysis')
            filename = f"fit_indices_{analysis_type}_{timestamp}.csv"
        
        file_path = self.output_dir / filename
        
        # 적합도 지수를 DataFrame으로 변환
        fit_data = []
        for index_name, value in results['fit_indices'].items():
            fit_data.append({
                'Fit_Index': index_name,
                'Value': value,
                'Analysis_Type': results.get('analysis_type', 'unknown'),
                'Sample_Size': results.get('model_info', {}).get('n_observations', 'unknown')
            })
        
        fit_df = pd.DataFrame(fit_data)
        
        # 적합도 해석 추가
        fit_df['Interpretation'] = fit_df.apply(self._interpret_fit_index, axis=1)
        
        # CSV 저장
        fit_df.to_csv(file_path, index=False, encoding='utf-8-sig')
        logger.info(f"적합도 지수 저장 완료: {file_path}")
        
        return file_path
    
    def export_standardized_loadings(self, results: Dict[str, Any], 
                                   filename: Optional[str] = None) -> Path:
        """
        표준화된 factor loadings를 CSV 파일로 내보내기
        
        Args:
            results (Dict[str, Any]): 분석 결과
            filename (Optional[str]): 파일명
            
        Returns:
            Path: 저장된 파일 경로
        """
        if 'standardized_results' not in results or results['standardized_results'].empty:
            raise ValueError("표준화 결과 데이터가 없습니다")
        
        # 파일명 생성
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_type = results.get('analysis_type', 'factor_analysis')
            filename = f"standardized_loadings_{analysis_type}_{timestamp}.csv"
        
        file_path = self.output_dir / filename
        
        # 표준화 결과 테이블 준비
        std_df = results['standardized_results'].copy()
        
        # 추가 정보
        std_df['Analysis_Type'] = results.get('analysis_type', 'unknown')
        std_df['Sample_Size'] = results.get('model_info', {}).get('n_observations', 'unknown')
        
        # 표준화 계수 해석
        std_df['Loading_Strength'] = std_df['Std_Loading'].apply(self._interpret_loading_strength)
        
        # CSV 저장
        std_df.to_csv(file_path, index=False, encoding='utf-8-sig')
        logger.info(f"표준화 loadings 저장 완료: {file_path}")
        
        return file_path
    
    def export_comprehensive_results(self, results: Dict[str, Any], 
                                   base_filename: Optional[str] = None) -> Dict[str, Path]:
        """
        모든 결과를 종합적으로 내보내기
        
        Args:
            results (Dict[str, Any]): 분석 결과
            base_filename (Optional[str]): 기본 파일명
            
        Returns:
            Dict[str, Path]: 저장된 파일들의 경로
        """
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_type = results.get('analysis_type', 'factor_analysis')
            base_filename = f"factor_analysis_{analysis_type}_{timestamp}"
        
        saved_files = {}
        
        # Factor loadings
        try:
            loadings_file = self.export_factor_loadings(
                results, f"{base_filename}_loadings.csv"
            )
            saved_files['factor_loadings'] = loadings_file
        except Exception as e:
            logger.warning(f"Factor loadings 저장 실패: {e}")
        
        # 적합도 지수
        try:
            fit_file = self.export_fit_indices(
                results, f"{base_filename}_fit_indices.csv"
            )
            saved_files['fit_indices'] = fit_file
        except Exception as e:
            logger.warning(f"적합도 지수 저장 실패: {e}")
        
        # 표준화 결과
        try:
            std_file = self.export_standardized_loadings(
                results, f"{base_filename}_standardized.csv"
            )
            saved_files['standardized_loadings'] = std_file
        except Exception as e:
            logger.warning(f"표준화 결과 저장 실패: {e}")
        
        # 요약 보고서
        try:
            summary_file = self.export_summary_report(
                results, f"{base_filename}_summary.txt"
            )
            saved_files['summary_report'] = summary_file
        except Exception as e:
            logger.warning(f"요약 보고서 저장 실패: {e}")
        
        # 메타데이터
        try:
            metadata_file = self.export_metadata(
                results, f"{base_filename}_metadata.json"
            )
            saved_files['metadata'] = metadata_file
        except Exception as e:
            logger.warning(f"메타데이터 저장 실패: {e}")
        
        logger.info(f"종합 결과 저장 완료: {len(saved_files)}개 파일")
        return saved_files
    
    def export_summary_report(self, results: Dict[str, Any], 
                            filename: Optional[str] = None) -> Path:
        """
        요약 보고서를 텍스트 파일로 내보내기
        
        Args:
            results (Dict[str, Any]): 분석 결과
            filename (Optional[str]): 파일명
            
        Returns:
            Path: 저장된 파일 경로
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"factor_analysis_summary_{timestamp}.txt"
        
        file_path = self.output_dir / filename
        
        # 요약 보고서 생성
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("FACTOR ANALYSIS RESULTS SUMMARY")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 모델 정보
        model_info = results.get('model_info', {})
        report_lines.append("MODEL INFORMATION")
        report_lines.append("-" * 30)
        report_lines.append(f"Sample Size: {model_info.get('n_observations', 'N/A')}")
        report_lines.append(f"Variables: {model_info.get('n_variables', 'N/A')}")
        report_lines.append(f"Estimator: {model_info.get('estimator', 'N/A')}")
        report_lines.append(f"Analysis Type: {results.get('analysis_type', 'N/A')}")
        report_lines.append("")
        
        # 적합도 지수
        fit_indices = results.get('fit_indices', {})
        if fit_indices:
            report_lines.append("FIT INDICES")
            report_lines.append("-" * 30)
            for index, value in fit_indices.items():
                interpretation = self._interpret_fit_index_value(index, value)
                report_lines.append(f"{index}: {value} ({interpretation})")
            report_lines.append("")
        
        # Factor loadings 요약
        loadings = results.get('factor_loadings', pd.DataFrame())
        if not loadings.empty:
            report_lines.append("FACTOR LOADINGS SUMMARY")
            report_lines.append("-" * 30)
            
            # 요인별 요약
            for factor in loadings['Factor'].unique():
                factor_loadings = loadings[loadings['Factor'] == factor]
                significant_count = sum(factor_loadings['Significant'])
                avg_loading = factor_loadings['Loading'].mean()
                
                report_lines.append(f"{factor}:")
                report_lines.append(f"  Items: {len(factor_loadings)}")
                report_lines.append(f"  Significant loadings: {significant_count}")
                report_lines.append(f"  Average loading: {avg_loading:.3f}")
                report_lines.append("")
        
        # 파일 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"요약 보고서 저장 완료: {file_path}")
        return file_path
    
    def export_metadata(self, results: Dict[str, Any], 
                       filename: Optional[str] = None) -> Path:
        """
        분석 메타데이터를 JSON 파일로 내보내기
        
        Args:
            results (Dict[str, Any]): 분석 결과
            filename (Optional[str]): 파일명
            
        Returns:
            Path: 저장된 파일 경로
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"factor_analysis_metadata_{timestamp}.json"
        
        file_path = self.output_dir / filename
        
        # 메타데이터 준비 (JSON 직렬화 가능한 형태로)
        metadata = {
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_type': results.get('analysis_type', 'unknown'),
            'model_info': results.get('model_info', {}),
            'fit_indices': results.get('fit_indices', {}),
            'factor_names': results.get('factor_names', results.get('factor_name', [])),
            'n_factors': len(results.get('factor_loadings', pd.DataFrame())['Factor'].unique()) if 'factor_loadings' in results else 0,
            'n_items': len(results.get('factor_loadings', pd.DataFrame())) if 'factor_loadings' in results else 0
        }
        
        # JSON 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"메타데이터 저장 완료: {file_path}")
        return file_path
    
    def _interpret_fit_index(self, row: pd.Series) -> str:
        """적합도 지수 해석"""
        return self._interpret_fit_index_value(row['Fit_Index'], row['Value'])
    
    def _interpret_fit_index_value(self, index_name: str, value: float) -> str:
        """적합도 지수 값 해석"""
        if index_name in ['CFI', 'TLI']:
            if value >= 0.95:
                return "Excellent"
            elif value >= 0.90:
                return "Good"
            else:
                return "Poor"
        elif index_name == 'RMSEA':
            if value <= 0.05:
                return "Excellent"
            elif value <= 0.08:
                return "Good"
            else:
                return "Poor"
        elif index_name == 'SRMR':
            if value <= 0.05:
                return "Excellent"
            elif value <= 0.08:
                return "Good"
            else:
                return "Poor"
        else:
            return "N/A"
    
    def _interpret_loading_strength(self, loading: float) -> str:
        """Loading 강도 해석"""
        abs_loading = abs(loading)
        if abs_loading >= 0.7:
            return "Strong"
        elif abs_loading >= 0.5:
            return "Moderate"
        elif abs_loading >= 0.3:
            return "Weak"
        else:
            return "Very Weak"


def export_factor_results(results: Dict[str, Any], 
                         output_dir: Optional[Union[str, Path]] = None,
                         comprehensive: bool = True) -> Union[Path, Dict[str, Path]]:
    """
    Factor analysis 결과를 내보내는 편의 함수
    
    Args:
        results (Dict[str, Any]): 분석 결과
        output_dir (Optional[Union[str, Path]]): 출력 디렉토리
        comprehensive (bool): 종합 결과 내보내기 여부
        
    Returns:
        Union[Path, Dict[str, Path]]: 저장된 파일 경로(들)
    """
    exporter = FactorResultsExporter(output_dir)
    
    if comprehensive:
        return exporter.export_comprehensive_results(results)
    else:
        return exporter.export_factor_loadings(results)
