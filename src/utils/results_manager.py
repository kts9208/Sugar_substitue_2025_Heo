#!/usr/bin/env python3
"""
결과 파일 버전 관리 시스템

이 모듈은 분석 결과 파일들의 버전 관리를 담당합니다:
1. 자동 아카이브 시스템
2. 최신 결과 관리
3. 버전 메타데이터 추적
4. 결과 파일 검색 및 복원

Author: Sugar Substitute Research Team
Date: 2025-09-18
"""

import os
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class ResultsManager:
    """결과 파일 버전 관리 클래스"""
    
    def __init__(self, base_dir: str = "."):
        """
        초기화
        
        Args:
            base_dir: 프로젝트 루트 디렉토리
        """
        self.base_dir = Path(base_dir)
        self.current_dir = self.base_dir / "results" / "current"
        self.archive_dir = self.base_dir / "results" / "archive"
        self.metadata_file = self.base_dir / "results" / "metadata.json"
        
        # 디렉토리 생성
        self._ensure_directories()
        
        # 메타데이터 로드
        self.metadata = self._load_metadata()
    
    def _ensure_directories(self):
        """필요한 디렉토리들 생성"""
        directories = [
            self.current_dir / "factor_analysis",
            self.current_dir / "path_analysis", 
            self.current_dir / "reliability_analysis",
            self.current_dir / "discriminant_validity",
            self.current_dir / "correlations",
            self.current_dir / "moderation_analysis",
            self.current_dir / "multinomial_logit",
            self.current_dir / "utility_function",
            self.current_dir / "visualizations",
            self.archive_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self) -> Dict:
        """메타데이터 로드"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"메타데이터 로드 실패: {e}")
        
        return {
            "versions": {},
            "latest": {},
            "created": datetime.now().isoformat()
        }
    
    def _save_metadata(self):
        """메타데이터 저장"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"메타데이터 저장 실패: {e}")
    
    def archive_current_results(self, analysis_type: str, 
                              description: str = "") -> str:
        """
        현재 결과를 아카이브로 이동
        
        Args:
            analysis_type: 분석 유형 (factor_analysis, path_analysis 등)
            description: 아카이브 설명
            
        Returns:
            아카이브 디렉토리 경로
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_subdir = self.archive_dir / f"{timestamp}_{analysis_type}"
        
        current_analysis_dir = self.current_dir / analysis_type
        
        if current_analysis_dir.exists() and any(current_analysis_dir.iterdir()):
            try:
                # 아카이브 디렉토리로 이동
                shutil.move(str(current_analysis_dir), str(archive_subdir))
                
                # 새로운 current 디렉토리 생성
                current_analysis_dir.mkdir(parents=True, exist_ok=True)
                
                # 메타데이터 업데이트
                version_info = {
                    "timestamp": timestamp,
                    "analysis_type": analysis_type,
                    "description": description,
                    "archived_path": str(archive_subdir),
                    "file_count": len(list(archive_subdir.rglob("*")))
                }
                
                if analysis_type not in self.metadata["versions"]:
                    self.metadata["versions"][analysis_type] = []
                
                self.metadata["versions"][analysis_type].append(version_info)
                self._save_metadata()
                
                logger.info(f"결과 아카이브 완료: {analysis_type} → {archive_subdir}")
                return str(archive_subdir)
                
            except Exception as e:
                logger.error(f"아카이브 실패: {e}")
                return ""
        else:
            logger.info(f"아카이브할 결과가 없음: {analysis_type}")
            return ""
    
    def save_results(self, analysis_type: str, results: Dict[str, Any],
                    files: Dict[str, str] = None, 
                    auto_archive: bool = True) -> Dict[str, str]:
        """
        분석 결과 저장
        
        Args:
            analysis_type: 분석 유형
            results: 분석 결과 딕셔너리
            files: 저장할 파일들 {파일명: 내용}
            auto_archive: 자동 아카이브 여부
            
        Returns:
            저장된 파일 경로들
        """
        # 기존 결과 아카이브 (옵션)
        if auto_archive:
            self.archive_current_results(analysis_type, "자동 아카이브")
        
        # 결과 저장 디렉토리
        save_dir = self.current_dir / analysis_type
        save_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 메인 결과 JSON 저장
            results_file = save_dir / f"results_{timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            saved_files["results"] = str(results_file)
            
            # 추가 파일들 저장
            if files:
                for filename, content in files.items():
                    file_path = save_dir / f"{filename}_{timestamp}"
                    
                    if isinstance(content, str):
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                    elif hasattr(content, 'to_csv'):  # DataFrame
                        content.to_csv(file_path.with_suffix('.csv'), 
                                     index=False, encoding='utf-8')
                    
                    saved_files[filename] = str(file_path)
            
            # 최신 결과 메타데이터 업데이트
            self.metadata["latest"][analysis_type] = {
                "timestamp": timestamp,
                "results_file": str(results_file),
                "saved_files": saved_files,
                "file_count": len(saved_files)
            }
            self._save_metadata()
            
            logger.info(f"결과 저장 완료: {analysis_type} ({len(saved_files)}개 파일)")
            return saved_files
            
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
            return {}
    
    def get_latest_results(self, analysis_type: str) -> Optional[Dict]:
        """
        최신 결과 정보 조회
        
        Args:
            analysis_type: 분석 유형
            
        Returns:
            최신 결과 정보 또는 None
        """
        return self.metadata["latest"].get(analysis_type)
    
    def list_versions(self, analysis_type: str) -> List[Dict]:
        """
        특정 분석 유형의 모든 버전 조회
        
        Args:
            analysis_type: 분석 유형
            
        Returns:
            버전 정보 리스트
        """
        return self.metadata["versions"].get(analysis_type, [])
    
    def restore_version(self, analysis_type: str, timestamp: str) -> bool:
        """
        특정 버전을 현재 결과로 복원
        
        Args:
            analysis_type: 분석 유형
            timestamp: 복원할 버전의 타임스탬프
            
        Returns:
            복원 성공 여부
        """
        versions = self.list_versions(analysis_type)
        target_version = None
        
        for version in versions:
            if version["timestamp"] == timestamp:
                target_version = version
                break
        
        if not target_version:
            logger.error(f"버전을 찾을 수 없음: {analysis_type} {timestamp}")
            return False
        
        try:
            # 현재 결과 백업
            self.archive_current_results(analysis_type, f"복원 전 백업 ({timestamp})")
            
            # 아카이브에서 복원
            archive_path = Path(target_version["archived_path"])
            current_path = self.current_dir / analysis_type
            
            if archive_path.exists():
                shutil.copytree(archive_path, current_path, dirs_exist_ok=True)
                logger.info(f"버전 복원 완료: {analysis_type} {timestamp}")
                return True
            else:
                logger.error(f"아카이브 파일을 찾을 수 없음: {archive_path}")
                return False
                
        except Exception as e:
            logger.error(f"버전 복원 실패: {e}")
            return False
    
    def cleanup_old_versions(self, analysis_type: str, keep_count: int = 5):
        """
        오래된 버전들 정리
        
        Args:
            analysis_type: 분석 유형
            keep_count: 유지할 버전 수
        """
        versions = self.list_versions(analysis_type)
        
        if len(versions) <= keep_count:
            return
        
        # 타임스탬프 기준 정렬 (최신순)
        versions.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # 오래된 버전들 제거
        versions_to_remove = versions[keep_count:]
        
        for version in versions_to_remove:
            try:
                archive_path = Path(version["archived_path"])
                if archive_path.exists():
                    shutil.rmtree(archive_path)
                    logger.info(f"오래된 버전 제거: {version['timestamp']}")
            except Exception as e:
                logger.error(f"버전 제거 실패: {e}")
        
        # 메타데이터 업데이트
        self.metadata["versions"][analysis_type] = versions[:keep_count]
        self._save_metadata()
    
    def get_summary(self) -> Dict:
        """결과 관리 현황 요약"""
        summary = {
            "total_analysis_types": len(self.metadata["latest"]),
            "latest_results": {},
            "version_counts": {},
            "total_archived_versions": 0
        }
        
        for analysis_type, latest_info in self.metadata["latest"].items():
            summary["latest_results"][analysis_type] = latest_info["timestamp"]
        
        for analysis_type, versions in self.metadata["versions"].items():
            summary["version_counts"][analysis_type] = len(versions)
            summary["total_archived_versions"] += len(versions)
        
        return summary


# 편의 함수들
def save_results(analysis_type: str, results: Dict[str, Any], 
                files: Dict[str, str] = None, auto_archive: bool = True) -> Dict[str, str]:
    """결과 저장 편의 함수"""
    manager = ResultsManager()
    return manager.save_results(analysis_type, results, files, auto_archive)


def archive_previous_results(analysis_type: str, description: str = "") -> str:
    """이전 결과 아카이브 편의 함수"""
    manager = ResultsManager()
    return manager.archive_current_results(analysis_type, description)


def get_latest_results(analysis_type: str) -> Optional[Dict]:
    """최신 결과 조회 편의 함수"""
    manager = ResultsManager()
    return manager.get_latest_results(analysis_type)


def list_all_versions(analysis_type: str) -> List[Dict]:
    """모든 버전 조회 편의 함수"""
    manager = ResultsManager()
    return manager.list_versions(analysis_type)
