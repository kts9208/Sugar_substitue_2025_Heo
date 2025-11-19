import pickle
from pathlib import Path

# PKL 파일 찾기
results_dir = Path('results')
pkl_files = list(results_dir.glob('simultaneous_*_20251119_175057.pkl'))

if pkl_files:
    pkl_file = pkl_files[0]
    print(f"PKL 파일: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        result = pickle.load(f)
    
    print("\n=== Result Dictionary Keys ===")
    print(list(result.keys()))
    
    print("\n=== 'parameter_statistics' in result ===")
    print('parameter_statistics' in result)
    
    if 'parameter_statistics' in result and result['parameter_statistics']:
        print("\n=== Parameter Statistics Keys ===")
        print(list(result['parameter_statistics'].keys()))
        
        if 'choice' in result['parameter_statistics']:
            print("\n=== Choice Statistics Keys ===")
            print(list(result['parameter_statistics']['choice'].keys()))
    else:
        print("\n❌ parameter_statistics가 없거나 비어있음")
    
    print("\n=== Parameters Keys ===")
    if 'parameters' in result:
        print(list(result['parameters'].keys()))
        
        if 'choice' in result['parameters']:
            print("\n=== Choice Parameters ===")
            choice = result['parameters']['choice']
            for key, value in choice.items():
                print(f"  {key}: {value}")
else:
    print("PKL 파일을 찾을 수 없습니다.")

