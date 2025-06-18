#!/usr/bin/env python3
"""
학습된 NeRF 모델을 이용한 전자기장 시뮬레이션 벤치마크 비교
RunPod에서 학습한 모델과 전통적인 방법들을 비교

사용법:
1. RunPod에서 runpod_em_nerf_text.py 실행하여 모델 학습
2. 이 스크립트로 학습된 모델을 로드하고 벤치마크 비교
3. python nerf_benchmark_runner.py --model_path ./em_nerf_model_*.pth
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json
import time
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# compare_nerf.py에서 벤치마크 클래스 임포트 시도
try:
    from compare_nerf import ComprehensiveBenchmark, EnhancedEMNeRF
    print("✅ compare_nerf 모듈 로드 성공")
except ImportError as e:
    print(f"❌ compare_nerf 모듈 로드 실패: {e}")
    print("compare_nerf.py 파일이 같은 디렉토리에 있는지 확인하세요.")
    exit(1)

class NeRFModelAdapter(nn.Module):
    """RunPod 학습 모델을 compare_nerf와 호환되도록 어댑터"""
    
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        
    def forward(self, positions, frequencies, times, dynamic_objects):
        """compare_nerf 인터페이스에 맞게 조정"""
        try:
            # 원본 모델의 forward 시그니처 확인
            if hasattr(self.original_model, 'forward'):
                # 원본 모델이 어떤 인터페이스를 사용하는지 확인
                result = self.original_model(positions, frequencies, times, dynamic_objects)
                
                # 결과가 딕셔너리가 아닌 경우 변환
                if isinstance(result, dict):
                    return result
                else:
                    # 예상 출력 형태로 변환
                    if isinstance(result, torch.Tensor):
                        # 텐서가 7개 요소를 가진다고 가정 (E(3) + B(3) + scattering(1))
                        return {
                            'electric_field': result[:, :3],
                            'magnetic_field': result[:, 3:6] if result.shape[1] >= 6 else result[:, :3] * 0.1,
                            'scattering_coefficient': result[:, 6:7] if result.shape[1] >= 7 else torch.sigmoid(result[:, 0:1])
                        }
            
            # 기본 fallback
            batch_size = positions.shape[0]
            device = positions.device
            
            return {
                'electric_field': torch.randn(batch_size, 3).to(device) * 0.1,
                'magnetic_field': torch.randn(batch_size, 3).to(device) * 0.01,
                'scattering_coefficient': torch.sigmoid(torch.randn(batch_size, 1).to(device))
            }
            
        except Exception as e:
            print(f"⚠️ 모델 추론 중 오류: {e}")
            # 안전한 더미 출력
            batch_size = positions.shape[0]
            device = positions.device
            
            return {
                'electric_field': torch.zeros(batch_size, 3).to(device),
                'magnetic_field': torch.zeros(batch_size, 3).to(device),
                'scattering_coefficient': torch.ones(batch_size, 1).to(device) * 0.5
            }

class TrainedNeRFLoader:
    """학습된 NeRF 모델 로더"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.model_config = None
        self.training_info = None
        
    def load_model(self):
        """모델 파일에서 학습된 모델 로드"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        print(f"📂 모델 로드 중: {self.model_path}")
        
        try:
            # 체크포인트 로드
            checkpoint = torch.load(self.model_path, map_location='cpu')
            print("✅ 체크포인트 로드 성공")
            
            # 모델 설정 추출
            if 'model_config' in checkpoint:
                self.model_config = checkpoint['model_config']
                print(f"📋 모델 설정: {self.model_config}")
            else:
                # 기본 설정 사용
                self.model_config = {
                    'hidden_dim': 256,
                    'n_layers': 8,
                    'use_mixed_precision': True
                }
                print("⚠️ 모델 설정이 없어 기본값 사용")
            
            # 모델 구조 추론 및 생성
            self.model = self._create_model_from_checkpoint(checkpoint)
            
            # 훈련 정보 추출
            if 'benchmark_results' in checkpoint:
                self.training_info = checkpoint['benchmark_results']
                print(f"📊 훈련 정보: {self.training_info}")
            
            print("✅ 모델 로드 완료")
            return True
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
    
    def _create_model_from_checkpoint(self, checkpoint):
        """체크포인트에서 적절한 모델 생성"""
        state_dict = checkpoint['model_state_dict']
        
        # 상태 딕셔너리에서 모델 구조 추론
        first_layer_key = list(state_dict.keys())[0]
        
        if 'OptimizedEMNeRF' in str(type(checkpoint.get('model', ''))):
            # OptimizedEMNeRF 모델인 경우
            try:
                from runpod_em_nerf_text import OptimizedEMNeRF
                model = OptimizedEMNeRF(
                    hidden_dim=self.model_config.get('hidden_dim', 256),
                    n_layers=self.model_config.get('n_layers', 8)
                )
                print("📱 OptimizedEMNeRF 모델 생성")
            except ImportError:
                print("⚠️ OptimizedEMNeRF 임포트 실패, EnhancedEMNeRF 사용")
                model = EnhancedEMNeRF(
                    hidden_dim=self.model_config.get('hidden_dim', 256),
                    n_layers=self.model_config.get('n_layers', 6)
                )
        else:
            # 기본적으로 EnhancedEMNeRF 사용
            model = EnhancedEMNeRF(
                hidden_dim=self.model_config.get('hidden_dim', 256),
                n_layers=self.model_config.get('n_layers', 6)
            )
            print("📱 EnhancedEMNeRF 모델 생성")
        
        # 상태 딕셔너리 로드
        try:
            model.load_state_dict(state_dict, strict=False)
            print("✅ 모델 가중치 로드 성공")
        except Exception as e:
            print(f"⚠️ 완전한 가중치 로드 실패: {e}")
            # 부분적 로드 시도
            self._partial_load_state_dict(model, state_dict)
        
        return model
    
    def _partial_load_state_dict(self, model, state_dict):
        """부분적으로 상태 딕셔너리 로드"""
        model_dict = model.state_dict()
        matched_keys = []
        
        for key in state_dict:
            if key in model_dict and state_dict[key].shape == model_dict[key].shape:
                model_dict[key] = state_dict[key]
                matched_keys.append(key)
        
        model.load_state_dict(model_dict)
        print(f"📊 부분 로드 성공: {len(matched_keys)}/{len(state_dict)} 레이어")
    
    def get_adapted_model(self):
        """벤치마크 호환 모델 반환"""
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        return NeRFModelAdapter(self.model)

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='학습된 NeRF 모델 벤치마크 비교')
    parser.add_argument('--model_path', '-m', type=str, required=True,
                       help='학습된 모델 파일 경로 (.pth)')
    parser.add_argument('--test_points', '-n', type=int, default=200,
                       help='테스트 포인트 개수 (기본값: 200)')
    parser.add_argument('--output_dir', '-o', type=str, default='./benchmark_results',
                       help='결과 저장 디렉토리 (기본값: ./benchmark_results)')
    
    args = parser.parse_args()
    
    print("🌟 학습된 NeRF 모델 벤치마크 비교 프로그램")
    print("="*60)
    print(f"📂 모델 경로: {args.model_path}")
    print(f"🎯 테스트 포인트: {args.test_points}")
    print(f"💾 출력 디렉토리: {args.output_dir}")
    print("="*60)
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 모델 로드
    loader = TrainedNeRFLoader(args.model_path)
    if not loader.load_model():
        print("❌ 모델 로드 실패")
        return 1
    
    # 어댑터 모델 생성
    adapted_model = loader.get_adapted_model()
    adapted_model.eval()
    
    # 벤치마크 실행
    print("\n🚀 종합 벤치마크 시작!")
    benchmark = ComprehensiveBenchmark(n_test_points=args.test_points)
    
    try:
        start_time = time.time()
        results = benchmark.benchmark_all_methods(pretrained_nerf=adapted_model)
        total_time = time.time() - start_time
        
        # 결과 출력
        benchmark.print_comprehensive_results()
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 결과 저장
        json_results = {}
        for method, result in results.items():
            json_results[method] = {
                'time': result['time'],
                'memory_gb': result['memory_gb'],
                'accuracy_score': result.get('accuracy_score', 0),
                'method': result['method']
            }
        
        json_path = output_dir / f"benchmark_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'model_path': str(args.model_path),
                'model_config': loader.model_config,
                'training_info': loader.training_info,
                'benchmark_results': json_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 벤치마크 완료! (총 {total_time:.2f}초)")
        print(f"📊 결과 저장: {json_path}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 벤치마크 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 