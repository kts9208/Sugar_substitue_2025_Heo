#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
역문항 처리 모듈

이 모듈은 설문 데이터의 역문항을 역코딩하여 처리합니다.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ReverseItemsProcessor:
    """역문항 처리 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        초기화
        
        Args:
            config_path: 역문항 설정 파일 경로 (기본값: data/config/reverse_items_config.json)
        """
        if config_path is None:
            config_path = "data/config/reverse_items_config.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # 척도 범위
        self.scale_min = self.config['scale_range']['min']
        self.scale_max = self.config['scale_range']['max']
        
        # 데이터 디렉토리
        self.data_dir = Path("data/processed/survey")
        
        logger.info(f"설정 파일 로드 완료: {self.config_path}")
        logger.info(f"ReverseItemsProcessor 초기화 완료")
        logger.info(f"척도 범위: {self.scale_min}-{self.scale_max}")
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"설정 파일 JSON 파싱 오류: {e}")
    
    def _reverse_code_value(self, value: float) -> float:
        """
        단일 값 역코딩
        
        Args:
            value: 원점수
        
        Returns:
            역코딩된 값
        """
        if pd.isna(value):
            return np.nan
        
        reversed_value = (self.scale_max + self.scale_min) - value
        return reversed_value
    
    def process_reverse_items(self, save_as_new: bool = True) -> bool:
        """
        전체 역문항 처리
        
        Args:
            save_as_new: True이면 새 파일로 저장 (*_reversed.csv), False이면 원본 덮어쓰기
        
        Returns:
            성공 여부
        """
        try:
            total_processed = 0
            total_errors = 0
            
            reverse_items_config = self.config['reverse_items']
            
            for factor_name, factor_config in reverse_items_config.items():
                reverse_items = factor_config.get('reverse_items', [])
                
                if not reverse_items:
                    logger.info(f"{factor_name}: 역문항이 없습니다.")
                    continue
                
                logger.info(f"{factor_name} 처리 시작...")
                
                # 데이터 파일 경로
                file_path = self.data_dir / f"{factor_name}.csv"
                
                if not file_path.exists():
                    logger.warning(f"{factor_name}: 데이터 파일을 찾을 수 없습니다: {file_path}")
                    total_errors += 1
                    continue
                
                # 데이터 로드
                data = pd.read_csv(file_path)
                
                # 역문항 처리
                for item in reverse_items:
                    if item not in data.columns:
                        logger.warning(f"{factor_name}.{item}: 컬럼을 찾을 수 없습니다")
                        total_errors += 1
                        continue
                    
                    # 역코딩
                    data[item] = data[item].apply(self._reverse_code_value)
                    total_processed += 1
                    logger.info(f"{factor_name}.{item} 역코딩 완료: {len(data)}개 값 처리")
                
                # 저장
                if save_as_new:
                    # 새 파일로 저장
                    output_path = self.data_dir / f"{factor_name}_reversed.csv"
                else:
                    # 원본 덮어쓰기
                    output_path = file_path
                
                data.to_csv(output_path, index=False)
                logger.info(f"{factor_name} 역문항 처리 완료 및 저장: {output_path}")
                logger.info(f"{factor_name} 처리 완료: {len(reverse_items)}개 문항")
            
            logger.info(f"전체 역문항 처리 완료: {total_processed}개 문항 처리, {total_errors}개 오류")
            
            return total_errors == 0
            
        except Exception as e:
            logger.error(f"역문항 처리 중 오류 발생: {e}")
            return False
    
    def verify_reverse_coding(self, factor_name: str) -> Dict[str, Any]:
        """
        역코딩 검증
        
        Args:
            factor_name: 요인 이름
        
        Returns:
            검증 결과
        """
        try:
            # 원본 파일과 역코딩 파일 로드
            original_path = self.data_dir / f"{factor_name}.csv"
            reversed_path = self.data_dir / f"{factor_name}_reversed.csv"
            
            if not original_path.exists():
                return {"success": False, "error": f"원본 파일 없음: {original_path}"}
            
            if not reversed_path.exists():
                return {"success": False, "error": f"역코딩 파일 없음: {reversed_path}"}
            
            original_data = pd.read_csv(original_path)
            reversed_data = pd.read_csv(reversed_path)
            
            # 역문항 목록
            factor_config = self.config['reverse_items'].get(factor_name, {})
            reverse_items = factor_config.get('reverse_items', [])
            
            if not reverse_items:
                return {"success": False, "error": "역문항이 없습니다"}
            
            # 검증 결과
            results = {}
            
            for item in reverse_items:
                if item not in original_data.columns or item not in reversed_data.columns:
                    results[item] = {"success": False, "error": "컬럼 없음"}
                    continue
                
                original_mean = original_data[item].mean()
                reversed_mean = reversed_data[item].mean()
                expected_mean = (self.scale_max + self.scale_min) - original_mean
                
                diff = abs(reversed_mean - expected_mean)
                
                results[item] = {
                    "success": diff < 0.001,
                    "original_mean": original_mean,
                    "reversed_mean": reversed_mean,
                    "expected_mean": expected_mean,
                    "diff": diff
                }
            
            return {
                "success": True,
                "factor_name": factor_name,
                "results": results
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_reverse_items_summary(self) -> pd.DataFrame:
        """
        역문항 요약 정보 반환
        
        Returns:
            역문항 요약 DataFrame
        """
        summary_data = []
        
        for factor_name, factor_config in self.config['reverse_items'].items():
            reverse_items = factor_config.get('reverse_items', [])
            
            summary_data.append({
                'factor': factor_name,
                'description': factor_config.get('description', ''),
                'total_items': factor_config.get('total_items', 0),
                'reverse_items_count': len(reverse_items),
                'reverse_items': ', '.join(reverse_items) if reverse_items else 'None'
            })
        
        return pd.DataFrame(summary_data)
    
    def print_summary(self):
        """역문항 요약 정보 출력"""
        print("=" * 80)
        print("역문항 처리 요약")
        print("=" * 80)
        
        summary_df = self.get_reverse_items_summary()
        
        print(f"\n총 요인 수: {len(summary_df)}")
        print(f"총 역문항 수: {summary_df['reverse_items_count'].sum()}")
        print(f"척도 범위: {self.scale_min}-{self.scale_max}")
        print(f"역코딩 공식: reversed = {self.scale_max + self.scale_min} - original")
        
        print("\n요인별 역문항:")
        print("-" * 80)
        
        for _, row in summary_df.iterrows():
            print(f"\n{row['factor']} ({row['description']})")
            print(f"  전체 문항: {row['total_items']}개")
            print(f"  역문항: {row['reverse_items_count']}개")
            if row['reverse_items_count'] > 0:
                print(f"  역문항 목록: {row['reverse_items']}")
        
        print("\n" + "=" * 80)


def main():
    """메인 실행 함수"""
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("역문항 처리 프로그램")
    print("=" * 80)
    
    # 프로세서 초기화
    processor = ReverseItemsProcessor()
    
    # 요약 정보 출력
    processor.print_summary()
    
    # 사용자 확인
    print("\n역문항 처리를 시작하시겠습니까?")
    print("  - 원본 파일은 유지됩니다")
    print("  - 역코딩된 데이터는 *_reversed.csv 파일로 저장됩니다")
    
    response = input("\n계속하시겠습니까? (y/n): ")
    
    if response.lower() != 'y':
        print("처리를 취소했습니다.")
        return
    
    # 역문항 처리 실행
    print("\n역문항 처리 시작...")
    print("-" * 80)
    
    success = processor.process_reverse_items(save_as_new=True)
    
    if success:
        print("\n✓ 역문항 처리 완료!")
        print(f"  저장 위치: {processor.data_dir}/*_reversed.csv")
    else:
        print("\n✗ 역문항 처리 중 오류가 발생했습니다.")
        print("  로그를 확인하세요.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

