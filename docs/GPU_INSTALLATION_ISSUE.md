# GPU 가속 설치 이슈 및 해결 방안

## 시스템 정보

- **GPU**: NVIDIA GeForce RTX 4060 (8GB VRAM)
- **CUDA**: 12.8
- **Driver**: 572.70
- **OS**: Windows 11
- **Python**: 3.11

## 설치 시도 결과

### CuPy 설치 실패

```bash
pip install cupy-cuda12x
```

**오류**:
```
OSError: [Errno 2] No such file or directory: 
'C:\\Users\\KimTaeseok\\AppData\\Local\\Packages\\...\\cupy\\_core\\include\\cupy\\_cccl\\libcudacxx\\cuda\\__ptx\\instructions\\generated\\fence_proxy_async_generic_sync_restrict.h'
```

**원인**: Windows 경로 길이 제한 (260자)
- OneDrive 경로가 매우 김: `C:\Users\KimTaeseok\OneDrive - 제주대학교\Research\Coding\...`
- CuPy 내부 파일 경로가 매우 김
- 합쳐서 260자 초과

## 해결 방안

### 방안 1: Windows 긴 경로 활성화 (권장)

1. **레지스트리 편집**
   ```
   Windows + R → regedit
   HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
   LongPathsEnabled = 1 (DWORD)
   ```

2. **그룹 정책 편집** (Windows Pro/Enterprise)
   ```
   gpedit.msc
   컴퓨터 구성 → 관리 템플릿 → 시스템 → 파일 시스템
   "Win32 긴 경로 사용" → 사용
   ```

3. **재부팅 후 재시도**
   ```bash
   pip install cupy-cuda12x
   ```

### 방안 2: 가상환경을 짧은 경로에 생성

```bash
# C 드라이브 루트에 가상환경 생성
cd C:\
mkdir venv
cd venv
python -m venv sugar_gpu

# 가상환경 활성화
C:\venv\sugar_gpu\Scripts\activate

# CuPy 설치
pip install cupy-cuda12x

# 프로젝트 패키지 설치
cd "C:\Users\KimTaeseok\OneDrive - 제주대학교\Research\Coding\Sugar_substitue_2025_Heo"
pip install -r requirements.txt
```

### 방안 3: Conda 사용

```bash
# Miniconda 설치 (C:\miniconda3)
# https://docs.conda.io/en/latest/miniconda.html

# 환경 생성
conda create -n sugar_gpu python=3.11
conda activate sugar_gpu

# CuPy 설치 (conda는 경로 문제 없음)
conda install -c conda-forge cupy cudatoolkit=12.0
```

### 방안 4: CPU 병렬처리 최적화 (현재 사용 중)

GPU 없이도 충분히 빠른 성능:
- **27개 CPU 코어** 병렬처리
- **예상 속도**: 순차 대비 ~20-27배
- **추정 시간**: 30-60분 (1000회 반복 기준)

## 권장 사항

### 즉시 실행 가능
✅ **방안 4: CPU 병렬처리로 모델 추정**
- 설치 불필요
- 이미 구현됨
- 충분히 빠름

### GPU 사용 원하는 경우
1. **방안 1** 시도 (가장 간단)
2. 실패 시 **방안 3** (Conda)
3. 최후 수단: **방안 2** (짧은 경로)

## 성능 비교 (예상)

| 방식 | 1회 우도 계산 | 1000회 반복 | 설치 난이도 |
|------|--------------|------------|------------|
| 순차 (1 CPU) | 60초 | 16.7시간 | - |
| **병렬 (27 CPU)** | **2.5초** | **42분** | ✅ **완료** |
| GPU (CuPy) | 1.5초 | 25분 | ❌ 설치 실패 |
| GPU (JAX) | 0.5초 | 8분 | ❌ 미설치 |

## 결론

**현재 상황**: CPU 병렬처리로 충분히 빠름 (27배 속도 향상)

**GPU 필요성**: 낮음
- CPU로 42분이면 충분히 빠름
- GPU 설치 복잡도 높음
- 성능 향상 폭 작음 (42분 → 25분)

**권장**: 
1. ✅ CPU 병렬처리로 모델 추정 완료
2. ⏸️ GPU는 나중에 필요 시 설치
3. 📊 결과 검증이 우선

## 다음 단계

1. CPU 병렬처리로 모델 추정 실행
2. 결과 확인 및 검증
3. 필요 시 GPU 설치 재시도

