# Windows 긴 경로 지원 활성화
# 관리자 권한으로 실행 필요

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host "Windows 긴 경로 지원 활성화" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan

# 관리자 권한 확인
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "`n❌ 관리자 권한이 필요합니다!" -ForegroundColor Red
    Write-Host "PowerShell을 관리자 권한으로 다시 실행하세요.`n" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n✅ 관리자 권한 확인됨`n" -ForegroundColor Green

# 레지스트리 경로
$regPath = "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem"
$regName = "LongPathsEnabled"

try {
    # 현재 값 확인
    $currentValue = Get-ItemProperty -Path $regPath -Name $regName -ErrorAction SilentlyContinue
    
    if ($currentValue.$regName -eq 1) {
        Write-Host "✅ 긴 경로 지원이 이미 활성화되어 있습니다.`n" -ForegroundColor Green
    } else {
        Write-Host "긴 경로 지원을 활성화합니다...`n" -ForegroundColor Yellow
        
        # 레지스트리 값 설정
        Set-ItemProperty -Path $regPath -Name $regName -Value 1 -Type DWord
        
        Write-Host "✅ 긴 경로 지원이 활성화되었습니다!`n" -ForegroundColor Green
        Write-Host "⚠️  변경사항을 적용하려면 시스템을 재부팅해야 합니다.`n" -ForegroundColor Yellow
    }
    
    # Git 긴 경로 지원도 활성화
    Write-Host "Git 긴 경로 지원 확인 중...`n" -ForegroundColor Yellow
    
    $gitConfig = git config --global core.longpaths 2>$null
    
    if ($gitConfig -eq "true") {
        Write-Host "✅ Git 긴 경로 지원이 이미 활성화되어 있습니다.`n" -ForegroundColor Green
    } else {
        git config --global core.longpaths true 2>$null
        if ($?) {
            Write-Host "✅ Git 긴 경로 지원이 활성화되었습니다!`n" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Git이 설치되어 있지 않거나 설정에 실패했습니다.`n" -ForegroundColor Yellow
        }
    }
    
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 69) -ForegroundColor Cyan
    Write-Host "완료!" -ForegroundColor Green
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 69) -ForegroundColor Cyan
    
    Write-Host "`n다음 단계:" -ForegroundColor Yellow
    Write-Host "  1. 시스템 재부팅 (권장)" -ForegroundColor White
    Write-Host "  2. 재부팅 후 CuPy 설치:" -ForegroundColor White
    Write-Host "     pip install cupy-cuda12x`n" -ForegroundColor Cyan
    
} catch {
    Write-Host "`n❌ 오류 발생: $_`n" -ForegroundColor Red
    exit 1
}

