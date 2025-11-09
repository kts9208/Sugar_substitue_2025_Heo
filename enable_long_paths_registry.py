"""
Windows 긴 경로 지원 활성화 (레지스트리 수정)
관리자 권한 필요
"""

import winreg
import sys

def enable_long_paths():
    """Windows 레지스트리에서 긴 경로 지원 활성화"""
    
    print("=" * 70)
    print("Windows 긴 경로 지원 활성화")
    print("=" * 70)
    
    try:
        # 레지스트리 키 열기
        key_path = r"SYSTEM\CurrentControlSet\Control\FileSystem"
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            key_path,
            0,
            winreg.KEY_READ | winreg.KEY_WRITE
        )
        
        # 현재 값 확인
        try:
            current_value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
            print(f"\n현재 LongPathsEnabled 값: {current_value}")
        except FileNotFoundError:
            current_value = 0
            print("\nLongPathsEnabled 값이 설정되지 않음 (기본값: 0)")
        
        if current_value == 1:
            print("✅ 긴 경로 지원이 이미 활성화되어 있습니다!")
        else:
            # 값 설정
            winreg.SetValueEx(key, "LongPathsEnabled", 0, winreg.REG_DWORD, 1)
            print("✅ 긴 경로 지원이 활성화되었습니다!")
            print("\n⚠️  변경사항을 적용하려면 시스템을 재부팅해야 합니다.")
        
        winreg.CloseKey(key)
        
        print("\n" + "=" * 70)
        print("완료!")
        print("=" * 70)
        
        return True
        
    except PermissionError:
        print("\n❌ 오류: 관리자 권한이 필요합니다!")
        print("   VSCode를 관리자 권한으로 실행하세요.")
        return False
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = enable_long_paths()
    sys.exit(0 if success else 1)

