"""
로그 파일 파싱 테스트 스크립트
"""
import re
from pathlib import Path

# 로그 파일 경로
project_root = Path(__file__).parent.parent
log_file = project_root / 'results' / 'gpu_batch_iclv_estimation_log.txt'

if log_file.exists():
    print(f"로그 파일 찾음: {log_file}")
    print(f"파일 크기: {log_file.stat().st_size / 1024:.2f} KB")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"파일 내용 길이: {len(content)} 문자")
    
    # "Parameter Scaling Comparison" 섹션 찾기
    # 타임스탬프 포함 패턴
    # 두 번째 ---- 라인 이후부터 세 번째 ---- 라인까지
    pattern = r'Parameter Scaling Comparison:.*?-{80}.*?-{80}\n(.*?)-{80}'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        print("\n✅ Parameter Scaling Comparison 섹션 찾음!")
        param_section = match.group(1)
        print(f"섹션 길이: {len(param_section)} 문자")
        print(f"\n섹션 내용 (처음 500자):\n{param_section[:500]}")
        
        # 각 파라미터 라인 파싱
        # 형식: 2025-11-12 17:46:30 - ζ_health_concern_q7                1.821545     1.821545     1.000000
        param_pattern = r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+-\s+([ζσ²γβλ_][^\s]+)\s+([-+]?[\d.]+)\s+([-+]?[\d.]+)\s+([-+]?[\d.]+)'
        
        param_list = []
        for line in param_section.strip().split('\n'):
            param_match = re.match(param_pattern, line.strip())
            if param_match:
                param_name = param_match.group(1)
                external_value = float(param_match.group(2))
                
                param_list.append({
                    'Coefficient': param_name,
                    'Estimate': external_value
                })
        
        print(f"\n✅ {len(param_list)}개 파라미터 파싱 완료")
        
        # 처음 10개와 마지막 10개 출력
        print("\n처음 10개 파라미터:")
        print("-" * 80)
        for i, param in enumerate(param_list[:10]):
            print(f"{i+1:2d}. {param['Coefficient']:45s} = {param['Estimate']:12.6f}")
        
        print("\n...")
        
        print("\n마지막 10개 파라미터:")
        print("-" * 80)
        for i, param in enumerate(param_list[-10:], start=len(param_list)-9):
            print(f"{i:2d}. {param['Coefficient']:45s} = {param['Estimate']:12.6f}")
        
        # 구조모델 파라미터 확인
        print("\n구조모델 파라미터:")
        print("-" * 80)
        for param in param_list:
            if param['Coefficient'].startswith('γ_'):
                print(f"  {param['Coefficient']:45s} = {param['Estimate']:12.6f}")
        
        # 선택모델 파라미터 확인
        print("\n선택모델 파라미터:")
        print("-" * 80)
        for param in param_list:
            if param['Coefficient'].startswith('β_') or param['Coefficient'].startswith('λ_'):
                print(f"  {param['Coefficient']:45s} = {param['Estimate']:12.6f}")
    else:
        print("\n❌ Parameter Scaling Comparison 섹션을 찾을 수 없습니다.")
        
        # 디버깅: "Parameter Scaling" 문자열 검색
        if 'Parameter Scaling' in content:
            print("  'Parameter Scaling' 문자열은 존재합니다.")
            idx = content.find('Parameter Scaling')
            print(f"  위치: {idx}")
            print(f"  주변 내용:\n{content[idx:idx+200]}")
        else:
            print("  'Parameter Scaling' 문자열이 없습니다.")
else:
    print(f"❌ 로그 파일을 찾을 수 없습니다: {log_file}")

