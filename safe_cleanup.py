#!/usr/bin/env python3
"""
안전한 정리 스크립트

중요한 결과를 아카이브한 후 불필요한 파일들을 안전하게 정리합니다.
"""

import shutil
from pathlib import Path
from datetime import datetime
import json

def create_final_archive():
    """최종 아카이브 생성"""
    print("💾 중요 결과 최종 아카이브")
    print("=" * 50)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path(f"results/archive/final_legacy_backup_{timestamp}")
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # 아카이브할 중요 결과 디렉토리
    important_dirs = [
        "final_english_charts",
        "discriminant_validity_results", 
        "factor_correlations_results",
        "rpl_analysis_results",
        "moderation_analysis_results",
        "path_analysis_results"
    ]
    
    archived_count = 0
    for dir_name in important_dirs:
        source_dir = Path(dir_name)
        if source_dir.exists():
            try:
                dest_dir = archive_dir / dir_name
                shutil.copytree(source_dir, dest_dir)
                print(f"   ✅ {dir_name} → 아카이브")
                archived_count += 1
            except Exception as e:
                print(f"   ❌ {dir_name}: {e}")
    
    print(f"\n📦 아카이브 완료: {archived_count}개 디렉토리")
    print(f"📁 위치: {archive_dir}")
    
    return archive_dir


def cleanup_old_scripts():
    """구 실행 스크립트 정리"""
    print(f"\n🧹 구 실행 스크립트 정리")
    print("-" * 40)
    
    old_scripts = [
        "run_all_moderation_combinations.py",
        "run_analysis_original_data.py", 
        "run_correlation_visualization.py",
        "run_discriminant_validity_analysis.py",
        "run_factor_analysis.py",  # 새 버전은 scripts/에 있음
        "run_factor_visualization.py",
        "run_four_factor_moderation_analysis.py",
        "run_moderation_analysis.py",
        "run_path_visualization.py",
        "run_reliability_analysis.py",  # 새 버전은 scripts/에 있음
        "run_reverse_items_processing.py",
        "run_semopy_correlations.py",
        "run_semopy_native_visualization.py",
        "run_utility_analysis.py"
    ]
    
    removed_count = 0
    for script in old_scripts:
        script_path = Path(script)
        if script_path.exists():
            try:
                script_path.unlink()
                print(f"   ✅ 삭제: {script}")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ 실패: {script} - {e}")
    
    print(f"\n📊 스크립트 정리: {removed_count}개 삭제")
    return removed_count


def cleanup_result_directories():
    """결과 디렉토리 정리"""
    print(f"\n🗂️ 결과 디렉토리 정리")
    print("-" * 40)
    
    result_dirs = [
        "comparison_analysis_results",
        "comprehensive_mediation_results",
        "discriminant_validity_results",
        "factor_analysis_results", 
        "factor_correlations_results",
        "moderation_analysis_results",
        "path_analysis_effects_test_results",
        "path_analysis_results",
        "real_data_effects_test_results",
        "reliability_analysis_results",
        "rpl_analysis_results",
        "simple_effects_test_results",
        "utility_function_analysis_results",
        "utility_function_results"
    ]
    
    removed_count = 0
    for dir_name in result_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"   ✅ 삭제: {dir_name}")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ 실패: {dir_name} - {e}")
    
    print(f"\n📊 결과 디렉토리 정리: {removed_count}개 삭제")
    return removed_count


def cleanup_visualization_dirs():
    """시각화 디렉토리 정리"""
    print(f"\n🎨 시각화 디렉토리 정리")
    print("-" * 40)
    
    viz_dirs = [
        "correlation_visualization_results",
        "factor_analysis_visualization_results", 
        "integrated_visualization_results",
        "semopy_native_visualization_results",
        "test_path_visualizations",
        "test_visualizations"
    ]
    
    removed_count = 0
    for dir_name in viz_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"   ✅ 삭제: {dir_name}")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ 실패: {dir_name} - {e}")
    
    print(f"\n📊 시각화 디렉토리 정리: {removed_count}개 삭제")
    return removed_count


def cleanup_misc_files():
    """기타 파일 정리"""
    print(f"\n📄 기타 파일 정리")
    print("-" * 40)
    
    misc_files = [
        "cleanup_duplicates.py",
        "compare_5factor_vs_4factor_models.py",
        "comprehensive_mediation_analysis.py",
        "comprehensive_moderation_test.py", 
        "comprehensive_path_analysis.py",
        "create_5factor_visualization.py",
        "create_comprehensive_path_model.py",
        "discriminant_validity_comparison_20250906_182455.png",
        "example_visualization_usage.py",
        "final_test_structural_paths.py",
        "real_data_effects_test.py",
        "rpl_analysis_final.py",
        "rpl_corrected_coefficient_summary.py",
        "rpl_utility_function.py",
        "sem_effect_calculation_analysis.py",
        "test_heatmap.png",
        "utility_function_analysis.py",
        "manual_health_concern.dot"
    ]
    
    removed_count = 0
    for file_name in misc_files:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"   ✅ 삭제: {file_name}")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ 실패: {file_name} - {e}")
    
    print(f"\n📊 기타 파일 정리: {removed_count}개 삭제")
    return removed_count


def cleanup_log_reports():
    """로그 및 보고서 파일 정리"""
    print(f"\n📋 로그/보고서 파일 정리")
    print("-" * 40)
    
    log_files = [
        "discriminant_validity_comparison_report.py",
        "discriminant_validity_comparison_report_20250906_182455.txt",
        "path_analysis.log",
        "reverse_items_processing_report_20250905_134025.txt",
        "reverse_items_processing_report_20250905_144509.txt",
        "reverse_items_report_20250905_134110.txt",
        "reverse_items_report_20250905_140845.txt",
        "reverse_items_report_20250905_140952.txt",
        "reverse_items_report_20250905_141052.txt",
        "reverse_items_report_20250905_141219.txt",
        "reverse_items_report_20250905_143317.txt",
        "rpl_analysis_report.py",
        "significant_mediations_report_20250908_221854.txt"
    ]
    
    removed_count = 0
    for file_name in log_files:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"   ✅ 삭제: {file_name}")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ 실패: {file_name} - {e}")
    
    print(f"\n📊 로그/보고서 정리: {removed_count}개 삭제")
    return removed_count


def cleanup_processed_data():
    """processed_data 정리"""
    print(f"\n📁 processed_data 정리")
    print("-" * 40)
    
    processed_dir = Path("processed_data")
    if not processed_dir.exists():
        print("   ⚠️ processed_data 디렉토리 없음")
        return 0
    
    # reverse_items_config.json만 유지
    keep_file = "reverse_items_config.json"
    
    removed_count = 0
    for item in processed_dir.iterdir():
        if item.name != keep_file:
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                print(f"   ✅ 삭제: processed_data/{item.name}")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ 실패: processed_data/{item.name} - {e}")
    
    print(f"\n📊 processed_data 정리: {removed_count}개 삭제")
    return removed_count


def cleanup_cache_files():
    """캐시 파일 정리"""
    print(f"\n🗂️ 캐시 파일 정리")
    print("-" * 40)
    
    removed_count = 0
    
    # __pycache__ 디렉토리들
    for pycache_dir in Path(".").rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            print(f"   ✅ 삭제: {pycache_dir}")
            removed_count += 1
        except Exception as e:
            print(f"   ❌ 실패: {pycache_dir} - {e}")
    
    print(f"\n📊 캐시 정리: {removed_count}개 삭제")
    return removed_count


def cleanup_temp_test_files():
    """임시 테스트 파일 정리"""
    print(f"\n🧪 임시 테스트 파일 정리")
    print("-" * 40)
    
    temp_files = [
        "test_analysis_functions.py",
        "test_interactive_menu.py",
        "test_main_direct.py",
        "test_main_menu.py",
        "final_system_test.py",
        "simple_factor_test.py",
        "cleanup_legacy_files.py",
        "identify_cleanup_targets.py",
        "cleanup_targets.json",
        "cleanup_targets.txt"
    ]
    
    removed_count = 0
    for file_name in temp_files:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"   ✅ 삭제: {file_name}")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ 실패: {file_name} - {e}")
    
    print(f"\n📊 임시 테스트 파일 정리: {removed_count}개 삭제")
    return removed_count


def generate_cleanup_summary(archive_dir, total_removed):
    """정리 요약 보고서 생성"""
    print(f"\n📋 정리 요약 보고서 생성")
    print("-" * 40)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path(f"docs/cleanup_summary_{timestamp}.md")
    
    summary_content = f"""# 코드베이스 정리 완료 보고서

## 정리 개요
- **실행 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **총 삭제 항목**: {total_removed}개
- **아카이브 위치**: {archive_dir}

## 정리된 항목
1. **구 실행 스크립트**: 14개 (run_*.py)
2. **결과 디렉토리**: 14개 (*_results/)
3. **시각화 디렉토리**: 6개 (*_visualization_*)
4. **로그/보고서**: 13개 (*.log, *_report_*)
5. **기타 파일**: 18개 (분석 스크립트, 이미지 등)
6. **processed_data**: 10개 (백업, 임시 파일 등)
7. **캐시 파일**: __pycache__ 디렉토리들
8. **임시 테스트 파일**: 10개 (test_*, cleanup_* 등)

## 보존된 구조
```
Sugar_substitue_2025_Heo/
├── main.py                 # ✅ 통합 실행 시스템
├── config.py               # ✅ 설정 관리
├── README.md               # ✅ 프로젝트 문서
├── Raw data/               # ✅ 원본 데이터 (보존)
├── data/                   # ✅ 새로운 데이터 구조
├── src/                    # ✅ 모듈화된 소스 코드
├── scripts/                # ✅ 통합 실행 스크립트
├── results/                # ✅ 결과 관리 시스템
├── docs/                   # ✅ 문서화
├── logs/                   # ✅ 로그 관리
├── tests/                  # ✅ 테스트 코드
└── notebooks/              # ✅ 분석 노트북
```

## 중요 결과 아카이브
- **위치**: {archive_dir}
- **내용**: 최종 영문 차트, 판별타당도, 요인상관, RPL 분석, 조절효과, 경로분석 결과

## 사용법
```bash
# 대화형 메뉴
python main.py

# 개별 분석
python main.py --factor
python main.py --reliability  
python main.py --path

# 전체 분석
python main.py --all

# 결과 확인
python main.py --results
```

## 다음 단계
1. 새로운 통합 시스템으로 분석 수행
2. 필요시 아카이브에서 과거 결과 참조
3. 지속적인 결과 관리 및 버전 추적

---
*Sugar Substitute Research - 코드베이스 정리 완료*
"""
    
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print(f"   ✅ 요약 보고서: {summary_file}")
    return summary_file


def main():
    """메인 정리 함수"""
    print("🎯 Sugar Substitute Research - 안전한 코드베이스 정리")
    print("=" * 70)
    
    # 1. 중요 결과 아카이브
    archive_dir = create_final_archive()
    
    # 2. 단계별 정리
    total_removed = 0
    
    total_removed += cleanup_old_scripts()
    total_removed += cleanup_result_directories()
    total_removed += cleanup_visualization_dirs()
    total_removed += cleanup_misc_files()
    total_removed += cleanup_log_reports()
    total_removed += cleanup_processed_data()
    total_removed += cleanup_cache_files()
    total_removed += cleanup_temp_test_files()
    
    # 3. 요약 보고서 생성
    summary_file = generate_cleanup_summary(archive_dir, total_removed)
    
    # 4. 최종 결과
    print(f"\n" + "=" * 70)
    print("🎉 코드베이스 정리 완료!")
    print("=" * 70)
    print(f"🗑️ 총 삭제 항목: {total_removed}개")
    print(f"📦 아카이브 위치: {archive_dir}")
    print(f"📋 요약 보고서: {summary_file}")
    
    print(f"\n✅ 새로운 깔끔한 구조:")
    print(f"   📄 main.py - 통합 실행 시스템")
    print(f"   📁 data/ - 체계적인 데이터 관리")
    print(f"   📁 src/ - 모듈화된 소스 코드")
    print(f"   📁 scripts/ - 통합 실행 스크립트")
    print(f"   📁 results/ - 버전 관리 결과")
    
    print(f"\n🚀 이제 다음 명령으로 분석을 시작하세요:")
    print(f"   python main.py")


if __name__ == "__main__":
    main()
