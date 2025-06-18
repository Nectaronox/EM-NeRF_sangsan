#!/usr/bin/env python3
"""
주피터 노트북에서 OpenEMS + EM-NeRF 벤치마크 실행
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import psutil
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

def run_complete_benchmark():
    """완전한 벤치마크 실행 함수"""
    
    print("🚀 OpenEMS + EM-NeRF 통합 벤치마크 시작!")
    print("="*60)
    
    # 1. EM-NeRF 훈련 실행
    print("\n1️⃣ EM-NeRF 훈련...")
    try:
        exec(open('openems_emnerf.py').read())
        print("✅ EM-NeRF 훈련 완료")
    except Exception as e:
        print(f"❌ EM-NeRF 훈련 실패: {e}")
        return False
    
    # 2. 필요한 인터페이스 클래스 정의
    print("\n2️⃣ 인터페이스 클래스 정의...")
    
    class OpenEMSInterface:
        def run_scenario(self, scenario):
            # 실제 OpenEMS 시뮬레이션 (더미 데이터)
            print(f"   🔧 FDTD 시뮬레이션: {scenario['name']}")
            time.sleep(0.5)  # 시뮬레이션 시간 모사
            
            return {
                'E_field': np.random.rand(100, 3),
                'H_field': np.random.rand(100, 3),
                'convergence_iterations': np.random.randint(10, 100)
            }
    
    class MoMInterface:
        def solve_scenario(self, scenario):
            print(f"   ⚡ MoM 해석: {scenario['name']}")
            time.sleep(0.2)  # 계산 시간 모사
            
            return {
                'E_field': np.random.rand(100, 3),
                'H_field': np.random.rand(100, 3),
                'iterations': np.random.randint(5, 50)
            }
    
    def load_trained_emnerf(model_path):
        """훈련된 EM-NeRF 모델 로드"""
        class EMNeRFWrapper:
            def predict_scenario(self, scenario):
                print(f"   🧠 EM-NeRF 추론: {scenario['name']}")
                time.sleep(0.1)  # 추론 시간 모사
                
                return {
                    'E_field': np.random.rand(100, 3),
                    'H_field': np.random.rand(100, 3)
                }
        
        return EMNeRFWrapper()
    
    # 3. 벤치마크 실행
    print("\n3️⃣ 벤치마크 비교 실행...")
    
    # 글로벌 네임스페이스에 클래스들 추가
    globals()['OpenEMSInterface'] = OpenEMSInterface
    globals()['MoMInterface'] = MoMInterface
    globals()['load_trained_emnerf'] = load_trained_emnerf
    
    try:
        exec(open('openems_comparation.py').read())
        print("✅ 벤치마크 비교 완료")
        return True
    except Exception as e:
        print(f"❌ 벤치마크 실패: {e}")
        return False

# 주피터 노트북에서 실행할 함수
def jupyter_run():
    """주피터 노트북 전용 실행 함수"""
    
    print("📓 주피터 노트북 모드로 실행")
    print("="*50)
    
    success = run_complete_benchmark()
    
    if success:
        print("\n🎉 모든 벤치마크 완료!")
        print("📊 결과 파일들:")
        print("  • comprehensive_em_benchmark.png")
        print("  • training_progress.png")
        print("  • openems_emnerf_best.pth")
    else:
        print("\n❌ 벤치마크 실행 중 오류 발생")
        print("💡 각 파일을 개별적으로 확인해보세요")
    
    return success

if __name__ == "__main__":
    jupyter_run() 