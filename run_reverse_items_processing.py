#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
역문항(역코딩) 처리 실행 스크립트

이 스크립트는 설문 데이터의 역문항을 자동으로 식별하고 역코딩 처리합니다.
- 설정 파일 기반 역문항 정보 관리
- 원본 데이터 자동 백업
- 처리 결과 보고서 생성
- 데이터 유효성 검증
"""

import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append('.')

from processed_data.modules.reverse_items_processor import ReverseItemsProcessor, process_reverse_items

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reverse_items_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """메인 실행 함수"""
    print('=' * 80)
    print('역문항(역코딩) 처리 실행')
    print('=' * 80)
    
    start_time = datetime.now()
    print(f'처리 시작 시간: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print()
    
    try:
        # 1. 설정 및 디렉토리 확인
        config_path = "processed_data/reverse_items_config.json"
        data_dir = "processed_data/survey_data"
        
        print(f"설정 파일: {config_path}")
        print(f"데이터 디렉토리: {data_dir}")
        print()
        
        if not os.path.exists(config_path):
            logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
            return
        
        if not os.path.exists(data_dir):
            logger.error(f"데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
            return
        
        # 2. 역문항 처리기 초기화
        print("역문항 처리기 초기화 중...")
        processor = ReverseItemsProcessor(config_path, data_dir)
        print("✓ 초기화 완료")
        print()
        
        # 3. 설정 정보 출력
        print("역문항 설정 정보:")
        print("-" * 50)
        
        reverse_items_config = processor.config['reverse_items']
        total_reverse_items = 0
        
        for factor_name, factor_config in reverse_items_config.items():
            reverse_items = factor_config.get('reverse_items', [])
            total_items = factor_config.get('total_items', 0)
            
            print(f"[{factor_name}]")
            print(f"  전체 문항: {total_items}개")
            print(f"  역문항: {len(reverse_items)}개 ({', '.join(reverse_items) if reverse_items else '없음'})")
            print(f"  설명: {factor_config.get('note', '')}")
            
            total_reverse_items += len(reverse_items)
        
        print(f"\n전체 역문항 수: {total_reverse_items}개")
        print()
        
        # 4. 사용자 확인
        response = input("역문항 처리를 진행하시겠습니까? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("처리가 취소되었습니다.")
            return
        
        print()
        
        # 5. 역문항 처리 실행
        print("역문항 처리 시작...")
        print("-" * 50)
        
        results = processor.process_all_factors()
        
        if 'error' in results:
            logger.error(f"처리 실패: {results['error']}")
            return
        
        print("✓ 역문항 처리 완료")
        print()
        
        # 6. 결과 요약 출력
        print("처리 결과 요약:")
        print("-" * 50)
        
        print(f"백업 성공: {'예' if results.get('backup_success', False) else '아니오'}")
        print(f"전체 요인 수: {results.get('total_factors', 0)}")
        print(f"처리된 역문항 수: {results.get('total_reverse_items_processed', 0)}")
        print(f"오류 수: {results.get('total_errors', 0)}")
        print()
        
        # 7. 요인별 상세 결과
        print("요인별 처리 결과:")
        print("-" * 50)
        
        factor_results = results.get('factor_results', {})
        for factor_name, factor_result in factor_results.items():
            if 'error' in factor_result:
                print(f"❌ {factor_name}: {factor_result['error']}")
            elif factor_result.get('processed', False):
                processed_count = factor_result.get('total_processed', 0)
                reverse_items = factor_result.get('reverse_items', [])
                print(f"✅ {factor_name}: {processed_count}개 문항 처리 ({', '.join(reverse_items)})")
                
                # 문항별 평균 변화 표시
                processed_items = factor_result.get('processed_items', [])
                for item_info in processed_items:
                    item = item_info['item']
                    orig_mean = item_info['original_mean']
                    rev_mean = item_info['reversed_mean']
                    print(f"   - {item}: {orig_mean:.3f} → {rev_mean:.3f}")
            else:
                message = factor_result.get('message', '처리 안됨')
                print(f"ℹ️  {factor_name}: {message}")
        
        print()
        
        # 8. 보고서 생성 및 저장
        print("처리 보고서 생성 중...")
        report = processor.generate_processing_report(results)
        
        report_file = f"reverse_items_processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ 보고서 저장: {report_file}")
        print()
        
        # 9. 완료 메시지
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"처리 완료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"총 소요 시간: {duration.total_seconds():.2f}초")
        print()
        print("=" * 80)
        print("역문항(역코딩) 처리가 성공적으로 완료되었습니다!")
        print("=" * 80)
        
        # 10. 주의사항 안내
        print("\n⚠️  주의사항:")
        print("- 원본 데이터는 processed_data/survey_data_backup/ 에 백업되었습니다.")
        print("- 역코딩된 데이터로 신뢰도 분석을 다시 실행하시기 바랍니다.")
        print("- 처리 결과는 위에 생성된 보고서 파일에서 확인할 수 있습니다.")
        
    except Exception as e:
        logger.error(f"처리 실행 중 오류 발생: {e}")
        print(f"\n❌ 오류 발생: {e}")
        print("자세한 내용은 로그 파일을 확인하세요: reverse_items_processing.log")


def print_usage():
    """사용법 출력"""
    print("사용법:")
    print("  python run_reverse_items_processing.py")
    print()
    print("설명:")
    print("  설문 데이터의 역문항을 자동으로 식별하고 역코딩 처리합니다.")
    print()
    print("필요한 파일들:")
    print("  - processed_data/reverse_items_config.json (역문항 설정 파일)")
    print("  - processed_data/survey_data/*.csv (설문 데이터 파일들)")
    print()
    print("처리 결과:")
    print("  - 원본 데이터 백업 (processed_data/survey_data_backup/)")
    print("  - 역코딩된 데이터로 원본 파일 업데이트")
    print("  - 처리 보고서 생성 (reverse_items_processing_report_*.txt)")


def check_data_status():
    """현재 데이터 상태 확인"""
    print("현재 데이터 상태 확인:")
    print("-" * 50)
    
    try:
        from processed_data.modules.reverse_items_processor import ReverseItemsProcessor
        
        processor = ReverseItemsProcessor()
        config = processor.config['reverse_items']
        
        for factor_name, factor_config in config.items():
            data_file = processor.data_dir / f"{factor_name}.csv"
            if data_file.exists():
                import pandas as pd
                data = pd.read_csv(data_file)
                reverse_items = factor_config.get('reverse_items', [])
                
                print(f"[{factor_name}]")
                print(f"  파일: {data_file.name} ({data.shape[0]}행 × {data.shape[1]}열)")
                print(f"  역문항: {len(reverse_items)}개 ({', '.join(reverse_items) if reverse_items else '없음'})")
                
                # 역문항 평균값 확인 (역코딩 여부 추정)
                if reverse_items:
                    for item in reverse_items:
                        if item in data.columns:
                            mean_val = data[item].mean()
                            print(f"    - {item} 평균: {mean_val:.3f}")
                print()
            else:
                print(f"❌ {factor_name}: 파일 없음 ({data_file})")
        
    except Exception as e:
        print(f"상태 확인 중 오류: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help', 'help']:
            print_usage()
        elif sys.argv[1] in ['-s', '--status', 'status']:
            check_data_status()
        else:
            print("알 수 없는 옵션입니다. --help를 참조하세요.")
    else:
        main()
