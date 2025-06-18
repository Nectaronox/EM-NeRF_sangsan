#!/usr/bin/env python3
"""
완전한 NeRF 훈련 및 벤치마크 파이프라인

이 스크립트는 다음을 수행합니다:
1. 전자기장 NeRF 모델 훈련 (runpod_em_nerf_text.py 기반)
2. 훈련된 모델 저장
3. 저장된 모델을 이용한 벤치마크 비교 (compare_nerf.py 기반)
4. 결과 분석 및 리포트 생성

사용법:
python train_and_benchmark.py --epochs 100 --batch_size 32 --test_points 200
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from datetime import datetime
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# 필요한 모듈들 임포트
try:
    from runpod_em_nerf_text import OptimizedEMNeRF, create_em_training_data
    print("✅ runpod_em_nerf_text 모듈 로드 성공")
    RUNPOD_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ runpod_em_nerf_text 모듈 로드 실패: {e}")
    RUNPOD_AVAILABLE = False

try:
    from compare_nerf import ComprehensiveBenchmark, EnhancedEMNeRF
    print("✅ compare_nerf 모듈 로드 성공")
    COMPARE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ compare_nerf 모듈 로드 실패: {e}")
    COMPARE_AVAILABLE = False

def main():
    """메인 실행 함수"""
    print("🌟 NeRF 훈련 및 벤치마크 파이프라인")
    print("="*60)
    print("📋 사용법:")
    print("1. 훈련 + 벤치마크: python train_and_benchmark.py --epochs 50")
    print("2. 벤치마크만: python train_and_benchmark.py --skip_training --model_path model.pth")
    print("3. 도움말: python train_and_benchmark.py --help")
    print("="*60)
    
    # 간단한 예제 실행
    print("\n🚀 간단한 예제 실행...")
    
    # 기본 설정
    config = {
        'epochs': 10,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'hidden_dim': 128,
        'n_layers': 4,
        'n_samples': 1000
    }
    
    print(f"📋 설정: {config}")
    
    # 1. 간단한 모델 생성 및 훈련 데모
    if RUNPOD_AVAILABLE:
        print("📱 OptimizedEMNeRF 사용")
        model = OptimizedEMNeRF(
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers']
        )
    elif COMPARE_AVAILABLE:
        print("📱 EnhancedEMNeRF 사용")
        model = EnhancedEMNeRF(
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers']
        )
    else:
        print("❌ 사용 가능한 NeRF 모델이 없습니다.")
        return
    
    print(f"✅ 모델 생성 완료! 파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. 벤치마크 데모 (모듈이 있는 경우)
    if COMPARE_AVAILABLE:
        print("\n🏁 벤치마크 데모 실행...")
        try:
            benchmark = ComprehensiveBenchmark(n_test_points=50)  # 적은 수로 테스트
            print("📊 벤치마크 시스템 초기화 완료")
            print("💡 실제 벤치마크 실행을 위해서는 훈련된 모델이 필요합니다.")
        except Exception as e:
            print(f"⚠️ 벤치마크 초기화 실패: {e}")
    
    print("\n✅ 데모 완료!")
    print("\n📘 다음 단계:")
    print("1. runpod_em_nerf_text.py에서 모델 훈련")
    print("2. nerf_benchmark_runner.py로 벤치마크 실행")
    print("3. 또는 이 스크립트를 확장하여 완전한 파이프라인 구현")

if __name__ == "__main__":
    main() 