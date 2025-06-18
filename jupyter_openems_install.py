#!/usr/bin/env python3
"""
Jupyter 노트북용 EM-NeRF + OpenEMS 설치 스크립트
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """패키지 설치 함수"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} 설치 완료")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {package} 설치 실패")
        return False

def check_package(package):
    """패키지 설치 여부 확인"""
    try:
        __import__(package)
        print(f"✅ {package} 이미 설치됨")
        return True
    except ImportError:
        print(f"⚠️  {package} 설치 필요")
        return False

# 1. 기본 패키지 설치
print("🔧 기본 패키지 설치 중...")
basic_packages = [
    "numpy",
    "matplotlib", 
    "torch",
    "torchvision",
    "torchaudio",
    "h5py",
    "psutil",
    "tqdm",
    "scipy"
]

for package in basic_packages:
    if not check_package(package.split('==')[0]):
        install_package(package)

# 2. OpenEMS 관련 패키지 시도
print("\n🏗️ OpenEMS 인터페이스 설치 시도...")
openems_packages = [
    "openems-interface",
    "PyOpenEMS", 
    "python-openems"
]

openems_installed = False
for package in openems_packages:
    if install_package(package):
        openems_installed = True
        break

if not openems_installed:
    print("⚠️  OpenEMS Python 인터페이스 설치 실패")
    print("   가상 데이터 생성 모드로 실행됩니다")

# 3. 설치 확인
print("\n📋 설치 확인...")
try:
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import h5py
    print("✅ 모든 기본 패키지 정상 설치")
except ImportError as e:
    print(f"❌ 패키지 오류: {e}")

# OpenEMS 확인
try:
    # 여러 OpenEMS 인터페이스 시도
    packages_to_try = [
        ("pyems.simulation", "pyEMS"),
        ("openems", "OpenEMS Interface"), 
        ("PyOpenEMS", "PyOpenEMS")
    ]
    
    openems_available = False
    for package_name, display_name in packages_to_try:
        try:
            __import__(package_name)
            print(f"✅ {display_name} 사용 가능")
            openems_available = True
            break
        except ImportError:
            continue
    
    if not openems_available:
        print("⚠️  OpenEMS 인터페이스 없음 - 가상 데이터 모드")
        
except Exception as e:
    print(f"⚠️  OpenEMS 확인 중 오류: {e}")

print("\n🎉 설치 완료!")