#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
역문항 처리 실행 스크립트

이 스크립트는 설문 데이터의 역문항을 역코딩하여 별도 파일로 저장합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import pandas as pd
from src.data_processing.reverse_items_processor import ReverseItemsProcessor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/reverse_items_processing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("역문항 처리 프로그램")
    print("=" * 80)
    
    try:
        # 프로세서 초기화
        processor = ReverseItemsProcessor()
        
        # 요약 정보 출력
        processor.print_summary()
        
        # 역문항 처리 실행
        print("\n역문항 처리 시작...")
        print("-" * 80)
        print("  - 원본 파일: data/processed/survey/*.csv")
        print("  - 저장 파일: data/processed/survey/*_reversed.csv")
        print("-" * 80)
        
        success = processor.process_reverse_items(save_as_new=True)
        
        if success:
            print("\n✓ 역문항 처리 완료!")
            print(f"  저장 위치: {processor.data_dir}/*_reversed.csv")
            
            # 검증
            print("\n검증 시작...")
            print("-" * 80)
            
            factors_to_verify = ['perceived_benefit', 'perceived_price', 'nutrition_knowledge']
            
            for factor_name in factors_to_verify:
                result = processor.verify_reverse_coding(factor_name)
                
                if result['success']:
                    print(f"\n✓ {factor_name} 검증 완료:")
                    for item, item_result in result['results'].items():
                        if item_result['success']:
                            print(f"  ✓ {item}: {item_result['original_mean']:.2f} → {item_result['reversed_mean']:.2f}")
                        else:
                            print(f"  ✗ {item}: 검증 실패 (차이: {item_result['diff']:.4f})")
                else:
                    print(f"\n✗ {factor_name} 검증 실패: {result.get('error', 'Unknown error')}")
            
            # 파일 목록 출력
            print("\n생성된 파일:")
            print("-" * 80)
            
            reversed_files = list(processor.data_dir.glob("*_reversed.csv"))
            for file in sorted(reversed_files):
                file_size = file.stat().st_size
                print(f"  - {file.name} ({file_size:,} bytes)")
            
            print("\n" + "=" * 80)
            print("역문항 처리가 성공적으로 완료되었습니다!")
            print("=" * 80)
            
        else:
            print("\n✗ 역문항 처리 중 오류가 발생했습니다.")
            print("  로그를 확인하세요: logs/reverse_items_processing.log")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}", exc_info=True)
        print(f"\n✗ 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

