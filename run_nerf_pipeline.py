#!/usr/bin/env python3
"""
NeRF 모델 훈련 → 벤치마크 비교 완전 파이프라인

이 스크립트는 다음 단계를 순차적으로 실행합니다:
1. RunPod에서 전자기장 NeRF 모델 훈련
2. 훈련된 모델을 적절한 형태로 저장
3. 저장된 모델을 벤치마크 시스템에서 로드
4. 다른 전자기장 시뮬레이션 방법들과 성능 비교
5. 결과 분석 및 리포트 생성

사용법:
python run_nerf_pipeline.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

def check_dependencies():
    """필요한 파일들이 있는지 확인"""
    required_files = [
        'runpod_em_nerf_text.py',
        'compare_nerf.py',
        'nerf_benchmark_runner.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ 필요한 파일들이 없습니다:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ 모든 필요한 파일이 존재합니다.")
    return True

def step1_train_model():
    """1단계: NeRF 모델 훈련"""
    print("\n" + "="*60)
    print("🎯 1단계: NeRF 모델 훈련")
    print("="*60)
    
    print("🚀 runpod_em_nerf_text.py 실행 중...")
    
    try:
        # RunPod 스크립트 실행
        result = subprocess.run([
            sys.executable, 'runpod_em_nerf_text.py'
        ], capture_output=True, text=True, timeout=1800)  # 30분 타임아웃
        
        if result.returncode == 0:
            print("✅ 모델 훈련 완료!")
            print("📋 훈련 로그:")
            print(result.stdout)
            
            # 생성된 모델 파일 찾기
            model_files = list(Path(".").glob("em_nerf_model_*.pth"))
            if model_files:
                # 가장 최근 파일 반환
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                print(f"📂 훈련된 모델: {latest_model}")
                return str(latest_model)
            else:
                print("⚠️ 모델 파일을 찾을 수 없습니다. 수동으로 지정해주세요.")
                return None
        else:
            print("❌ 모델 훈련 실패!")
            print("오류 메시지:")
            print(result.stderr)
            return None
            
    except subprocess.TimeoutExpired:
        print("⏰ 훈련 시간이 너무 오래 걸립니다. 수동으로 확인해주세요.")
        return None
    except Exception as e:
        print(f"❌ 훈련 실행 중 오류: {e}")
        return None

def step2_run_benchmark(model_path):
    """2단계: 벤치마크 비교"""
    print("\n" + "="*60)
    print("🎯 2단계: 벤치마크 비교")
    print("="*60)
    
    if not model_path or not Path(model_path).exists():
        print("❌ 유효한 모델 파일이 없습니다.")
        return False
    
    print(f"📂 사용할 모델: {model_path}")
    print("🚀 nerf_benchmark_runner.py 실행 중...")
    
    try:
        # 벤치마크 실행
        result = subprocess.run([
            sys.executable, 'nerf_benchmark_runner.py',
            '--model_path', model_path,
            '--test_points', '100',  # 빠른 테스트를 위해 줄임
            '--output_dir', './pipeline_results'
        ], capture_output=True, text=True, timeout=600)  # 10분 타임아웃
        
        if result.returncode == 0:
            print("✅ 벤치마크 완료!")
            print("📋 벤치마크 결과:")
            print(result.stdout)
            return True
        else:
            print("❌ 벤치마크 실패!")
            print("오류 메시지:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 벤치마크 시간이 너무 오래 걸립니다.")
        return False
    except Exception as e:
        print(f"❌ 벤치마크 실행 중 오류: {e}")
        return False

def step3_generate_summary():
    """3단계: 결과 요약"""
    print("\n" + "="*60)
    print("🎯 3단계: 결과 요약")
    print("="*60)
    
    results_dir = Path('./pipeline_results')
    if not results_dir.exists():
        print("❌ 결과 디렉토리를 찾을 수 없습니다.")
        return
    
    # JSON 결과 파일 찾기
    json_files = list(results_dir.glob("benchmark_results_*.json"))
    if json_files:
        latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
        print(f"📊 최신 결과 파일: {latest_json}")
        
        try:
            import json
            with open(latest_json, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            print("\n📈 벤치마크 요약:")
            print("-" * 40)
            
            if 'benchmark_results' in results:
                for method, result in results['benchmark_results'].items():
                    print(f"🔹 {method}:")
                    print(f"   ⏱️  시간: {result['time']:.4f}초")
                    print(f"   💾 메모리: {result['memory_gb']:.3f}GB")
                    print(f"   🎯 정확도: {result['accuracy_score']:.3f}")
                    print()
            
            print("✅ 결과 요약 완료!")
            
        except Exception as e:
            print(f"⚠️ 결과 파일 읽기 실패: {e}")
    else:
        print("⚠️ 결과 파일을 찾을 수 없습니다.")

def main():
    """메인 실행 함수"""
    print("🌟 NeRF 전자기장 시뮬레이션 완전 파이프라인")
    print("="*60)
    print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📋 실행 단계:")
    print("   1️⃣ NeRF 모델 훈련 (runpod_em_nerf_text.py)")
    print("   2️⃣ 벤치마크 비교 (nerf_benchmark_runner.py)")
    print("   3️⃣ 결과 요약 및 분석")
    print("="*60)
    
    # 의존성 확인
    if not check_dependencies():
        print("\n❌ 필요한 파일들을 준비해주세요.")
        return 1
    
    start_time = time.time()
    
    try:
        # 1단계: 모델 훈련
        model_path = step1_train_model()
        
        if not model_path:
            print("\n⚠️ 모델 훈련이 완료되지 않았습니다.")
            print("💡 수동으로 모델을 훈련한 후 다음 명령으로 벤치마크를 실행하세요:")
            print("   python nerf_benchmark_runner.py --model_path your_model.pth")
            return 1
        
        # 2단계: 벤치마크
        benchmark_success = step2_run_benchmark(model_path)
        
        if not benchmark_success:
            print("\n⚠️ 벤치마크가 완료되지 않았습니다.")
            print("💡 수동으로 벤치마크를 실행해보세요:")
            print(f"   python nerf_benchmark_runner.py --model_path {model_path}")
            return 1
        
        # 3단계: 결과 요약
        step3_generate_summary()
        
        # 완료 메시지
        total_time = time.time() - start_time
        print(f"\n🎉 전체 파이프라인 완료!")
        print(f"⏱️ 총 실행 시간: {total_time:.1f}초")
        print(f"📂 결과 저장 위치: ./pipeline_results/")
        print(f"⏰ 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자가 실행을 중단했습니다.")
        return 1
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 