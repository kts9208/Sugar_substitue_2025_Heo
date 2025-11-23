import pickle

# 최신 부트스트랩 결과 로드
data = pickle.load(open('results/bootstrap/sequential/bootstrap_HC-PB_HC-PP_PB-PI_PP-PI_PI_NK_20251123_185639_full.pkl', 'rb'))

print('=' * 80)
print('부트스트랩 결과 요약')
print('=' * 80)
print()
print(f'성공한 샘플 수: {data["n_successful"]}')
print(f'실패한 샘플 수: {data["n_failed"]}')
print(f'총 샘플 수: {data["n_successful"] + data["n_failed"]}')
print(f'모드: {data["mode"]}')
print()
print(f'추정된 파라미터 수: {len(data["bootstrap_estimates"])}')
print(f'신뢰구간 계산된 파라미터 수: {len(data["confidence_intervals"])}')

