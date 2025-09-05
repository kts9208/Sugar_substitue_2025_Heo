"""
역문항(역코딩) 처리 모듈

이 모듈은 설문 데이터에서 역문항을 자동으로 식별하고 역코딩하는 기능을 제공합니다.
- 설정 파일 기반 역문항 정보 관리
- 자동 역코딩 처리
- 원본 데이터 백업 및 검증
- 처리 결과 보고서 생성
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)


class ReverseItemsProcessor:
    """역문항 처리 클래스"""
    
    def __init__(self, config_path: str = "processed_data/reverse_items_config.json",
                 data_dir: str = "processed_data/survey_data",
                 backup_dir: str = "processed_data/survey_data_backup"):
        """
        역문항 처리기 초기화
        
        Args:
            config_path (str): 역문항 설정 파일 경로
            data_dir (str): 설문 데이터 디렉토리
            backup_dir (str): 백업 디렉토리
        """
        self.config_path = Path(config_path)
        self.data_dir = Path(data_dir)
        self.backup_dir = Path(backup_dir)
        
        # 설정 로드
        self.config = self._load_config()
        self.scale_min = self.config['scale_range']['min']
        self.scale_max = self.config['scale_range']['max']
        
        # 백업 디렉토리 생성
        self.backup_dir.mkdir(exist_ok=True)
        
        logger.info(f"ReverseItemsProcessor 초기화 완료")
        logger.info(f"척도 범위: {self.scale_min}-{self.scale_max}")
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"설정 파일 로드 완료: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            raise
    
    def _reverse_code_value(self, value: float) -> float:
        """
        단일 값 역코딩
        
        Args:
            value (float): 원본 값
            
        Returns:
            float: 역코딩된 값
        """
        if pd.isna(value):
            return value
        
        # 역코딩 공식: reversed = (max + min) - original
        reversed_value = (self.scale_max + self.scale_min) - value
        return reversed_value
    
    def _validate_data(self, data: pd.DataFrame, factor_name: str) -> Tuple[bool, List[str]]:
        """
        데이터 유효성 검증
        
        Args:
            data (pd.DataFrame): 검증할 데이터
            factor_name (str): 요인명
            
        Returns:
            Tuple[bool, List[str]]: (유효성 여부, 오류 메시지 리스트)
        """
        errors = []
        
        # 1. 척도 범위 확인
        numeric_cols = [col for col in data.columns if col != 'no']
        for col in numeric_cols:
            values = data[col].dropna()
            if len(values) > 0:
                min_val = values.min()
                max_val = values.max()
                
                if min_val < self.scale_min or max_val > self.scale_max:
                    errors.append(f"{factor_name}.{col}: 값이 척도 범위({self.scale_min}-{self.scale_max})를 벗어남 (범위: {min_val}-{max_val})")
        
        # 2. 결측값 확인
        missing_counts = data.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0 and col != 'no':
                errors.append(f"{factor_name}.{col}: {count}개의 결측값 발견")
        
        return len(errors) == 0, errors
    
    def backup_original_data(self) -> bool:
        """
        원본 데이터 백업
        
        Returns:
            bool: 백업 성공 여부
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_subdir = self.backup_dir / f"backup_{timestamp}"
            backup_subdir.mkdir(exist_ok=True)
            
            # 모든 CSV 파일 백업
            csv_files = list(self.data_dir.glob("*.csv"))
            for csv_file in csv_files:
                backup_file = backup_subdir / csv_file.name
                shutil.copy2(csv_file, backup_file)
                logger.info(f"백업 완료: {csv_file.name}")
            
            logger.info(f"전체 데이터 백업 완료: {backup_subdir}")
            return True
            
        except Exception as e:
            logger.error(f"데이터 백업 실패: {e}")
            return False
    
    def process_factor_data(self, factor_name: str) -> Dict[str, Any]:
        """
        단일 요인 데이터의 역문항 처리
        
        Args:
            factor_name (str): 요인명
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            # 설정 정보 확인
            if factor_name not in self.config['reverse_items']:
                return {'error': f'요인 {factor_name}의 설정 정보를 찾을 수 없습니다.'}
            
            factor_config = self.config['reverse_items'][factor_name]
            reverse_items = factor_config.get('reverse_items', [])
            
            if not reverse_items:
                logger.info(f"{factor_name}: 역문항이 없습니다.")
                return {
                    'factor_name': factor_name,
                    'processed': False,
                    'reverse_items': [],
                    'message': '역문항 없음'
                }
            
            # 데이터 파일 로드
            data_file = self.data_dir / f"{factor_name}.csv"
            if not data_file.exists():
                return {'error': f'데이터 파일을 찾을 수 없습니다: {data_file}'}
            
            data = pd.read_csv(data_file)
            original_data = data.copy()
            
            # 데이터 유효성 검증
            is_valid, errors = self._validate_data(data, factor_name)
            if not is_valid:
                logger.warning(f"{factor_name} 데이터 검증 경고: {errors}")
            
            # 역문항 처리
            processed_items = []
            for item in reverse_items:
                if item in data.columns:
                    original_values = data[item].copy()
                    data[item] = data[item].apply(self._reverse_code_value)
                    
                    # 처리 통계
                    non_na_count = original_values.notna().sum()
                    processed_items.append({
                        'item': item,
                        'processed_count': non_na_count,
                        'original_mean': original_values.mean(),
                        'reversed_mean': data[item].mean(),
                        'original_std': original_values.std(),
                        'reversed_std': data[item].std()
                    })
                    
                    logger.info(f"{factor_name}.{item} 역코딩 완료: {non_na_count}개 값 처리")
                else:
                    logger.warning(f"{factor_name}: 역문항 {item}이 데이터에 없습니다.")
            
            # 처리된 데이터 저장
            data.to_csv(data_file, index=False)
            logger.info(f"{factor_name} 역문항 처리 완료 및 저장")
            
            return {
                'factor_name': factor_name,
                'processed': True,
                'reverse_items': reverse_items,
                'processed_items': processed_items,
                'total_processed': len(processed_items),
                'data_shape': data.shape,
                'validation_errors': errors if not is_valid else []
            }
            
        except Exception as e:
            logger.error(f"{factor_name} 역문항 처리 중 오류: {e}")
            return {'error': str(e)}
    
    def process_all_factors(self) -> Dict[str, Any]:
        """
        모든 요인의 역문항 처리
        
        Returns:
            Dict[str, Any]: 전체 처리 결과
        """
        try:
            # 원본 데이터 백업
            backup_success = self.backup_original_data()
            if not backup_success:
                logger.warning("데이터 백업에 실패했지만 처리를 계속합니다.")
            
            # 각 요인별 처리
            results = {}
            total_processed = 0
            total_errors = 0
            
            for factor_name in self.config['reverse_items'].keys():
                logger.info(f"{factor_name} 처리 시작...")
                result = self.process_factor_data(factor_name)
                results[factor_name] = result
                
                if 'error' in result:
                    total_errors += 1
                    logger.error(f"{factor_name} 처리 실패: {result['error']}")
                elif result.get('processed', False):
                    total_processed += result.get('total_processed', 0)
                    logger.info(f"{factor_name} 처리 완료: {result.get('total_processed', 0)}개 문항")
            
            # 전체 요약
            summary = {
                'timestamp': datetime.now().isoformat(),
                'backup_success': backup_success,
                'total_factors': len(self.config['reverse_items']),
                'total_reverse_items_processed': total_processed,
                'total_errors': total_errors,
                'factor_results': results
            }
            
            logger.info(f"전체 역문항 처리 완료: {total_processed}개 문항 처리, {total_errors}개 오류")
            return summary
            
        except Exception as e:
            logger.error(f"전체 역문항 처리 중 오류: {e}")
            return {'error': str(e)}
    
    def generate_processing_report(self, results: Dict[str, Any]) -> str:
        """
        처리 결과 보고서 생성
        
        Args:
            results (Dict[str, Any]): 처리 결과
            
        Returns:
            str: 보고서 텍스트
        """
        try:
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("역문항(역코딩) 처리 보고서")
            report_lines.append("=" * 80)
            report_lines.append("")
            
            # 전체 요약
            report_lines.append(f"처리 일시: {results.get('timestamp', 'N/A')}")
            report_lines.append(f"백업 성공: {'예' if results.get('backup_success', False) else '아니오'}")
            report_lines.append(f"전체 요인 수: {results.get('total_factors', 0)}")
            report_lines.append(f"처리된 역문항 수: {results.get('total_reverse_items_processed', 0)}")
            report_lines.append(f"오류 수: {results.get('total_errors', 0)}")
            report_lines.append("")
            
            # 요인별 상세 결과
            factor_results = results.get('factor_results', {})
            for factor_name, factor_result in factor_results.items():
                report_lines.append(f"[{factor_name}]")
                
                if 'error' in factor_result:
                    report_lines.append(f"  ❌ 오류: {factor_result['error']}")
                elif factor_result.get('processed', False):
                    report_lines.append(f"  ✅ 처리 완료")
                    report_lines.append(f"  - 역문항: {', '.join(factor_result.get('reverse_items', []))}")
                    report_lines.append(f"  - 처리된 문항 수: {factor_result.get('total_processed', 0)}")
                    
                    # 문항별 상세 정보
                    processed_items = factor_result.get('processed_items', [])
                    for item_info in processed_items:
                        item = item_info['item']
                        orig_mean = item_info['original_mean']
                        rev_mean = item_info['reversed_mean']
                        report_lines.append(f"    * {item}: {orig_mean:.3f} → {rev_mean:.3f}")
                else:
                    report_lines.append(f"  ℹ️  {factor_result.get('message', '처리 안됨')}")
                
                report_lines.append("")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"보고서 생성 중 오류: {e}")
            return f"보고서 생성 실패: {e}"


def process_reverse_items(config_path: str = "processed_data/reverse_items_config.json",
                         data_dir: str = "processed_data/survey_data") -> Dict[str, Any]:
    """
    역문항 처리 편의 함수
    
    Args:
        config_path (str): 설정 파일 경로
        data_dir (str): 데이터 디렉토리
        
    Returns:
        Dict[str, Any]: 처리 결과
    """
    processor = ReverseItemsProcessor(config_path, data_dir)
    return processor.process_all_factors()
