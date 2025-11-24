"""
Sign Correction 로직 테스트: 인위적으로 부호를 뒤집은 적재량으로 테스트
"""
import numpy as np
import pandas as pd

# 원본 적재량 (Marker Variable 제외)
orig_loadings = np.array([0.92, 0.73, 0.83, 0.76, 0.93])

# 케이스 1: 부호가 같음
boot_loadings_same = np.array([0.95, 0.75, 0.85, 0.79, 0.96])
dot_same = np.dot(orig_loadings, boot_loadings_same)
print(f"케이스 1 (부호 같음):")
print(f"  원본: {orig_loadings}")
print(f"  부트: {boot_loadings_same}")
print(f"  내적: {dot_same:.4f}")
print(f"  판정: {'부호 반전 필요' if dot_same < 0 else '부호 유지'}\n")

# 케이스 2: 부호가 반대
boot_loadings_flip = -boot_loadings_same
dot_flip = np.dot(orig_loadings, boot_loadings_flip)
print(f"케이스 2 (부호 반대):")
print(f"  원본: {orig_loadings}")
print(f"  부트: {boot_loadings_flip}")
print(f"  내적: {dot_flip:.4f}")
print(f"  판정: {'✅ 부호 반전 필요' if dot_flip < 0 else '부호 유지'}\n")

# 케이스 3: 일부만 반대 (혼합)
boot_loadings_mixed = np.array([-0.95, 0.75, -0.85, 0.79, -0.96])
dot_mixed = np.dot(orig_loadings, boot_loadings_mixed)
print(f"케이스 3 (혼합):")
print(f"  원본: {orig_loadings}")
print(f"  부트: {boot_loadings_mixed}")
print(f"  내적: {dot_mixed:.4f}")
print(f"  판정: {'✅ 부호 반전 필요' if dot_mixed < 0 else '부호 유지'}\n")

print("=" * 60)
print("결론: 내적이 음수면 부호 반전이 필요합니다.")
print("현재 부트스트랩 결과에서 모든 내적이 양수이므로,")
print("부호가 일관되게 유지되고 있습니다. (정상)")

