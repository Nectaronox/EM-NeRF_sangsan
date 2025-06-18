#!/usr/bin/env python3
"""
EM-NeRF vs FDTD vs MoM 종합 성능 비교
OpenEMS FDTD를 Ground Truth로 사용
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class BenchmarkResult:
    """벤치마크 결과 저장"""
    method_name: str
    accuracy: float          # MSE 기반 정확도
    computation_time: float  # 계산 시간 (초)
    memory_usage: float      # 메모리 사용량 (MB)
    convergence_rate: float  # 수렴 속도
    frequency_range: Tuple[float, float]
    scenario: str

class ComprehensiveEMBenchmark:
    """포괄적인 전자기 해석 방법 벤치마크"""
    
    def __init__(self, model):
        self.openems_fdtd = OpenEMSInterface()
        self.em_nerf = model # 훈련된 EM-NeRF 모델
        self.mom_solver = MoMInterface()
        self.results = []
    
    def setup_test_scenarios(self):
        """다양한 테스트 시나리오 정의"""
        return [
            {
                'name': 'simple_dipole',
                'complexity': 'low',
                'frequency': 2.4e9,
                'geometry': 'simple',
                'expected_advantage': 'analytical'  # 어떤 방법이 유리할지 예상
            },
            {
                'name': 'complex_array',
                'complexity': 'medium', 
                'frequency': 5.8e9,
                'geometry': 'moderate',
                'expected_advantage': 'fdtd'
            },
            {
                'name': 'urban_propagation',
                'complexity': 'high',
                'frequency': 28e9,
                'geometry': 'complex',
                'expected_advantage': 'nerf'  # 학습 기반이라 복잡한 환경에 유리
            },
            {
                'name': 'broadband_analysis', 
                'complexity': 'high',
                'frequency_range': (1e9, 10e9),
                'geometry': 'variable',
                'expected_advantage': 'fdtd'  # 광대역 분석
            }
        ]
    
    def run_fdtd_benchmark(self, scenario):
        """FDTD 방법 벤치마크 - Ground Truth 생성"""
        print(f"🔧 FDTD 시뮬레이션: {scenario['name']}")
        
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        # OpenEMS FDTD 실행
        fdtd_results = self.openems_fdtd.run_scenario(scenario)
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        return BenchmarkResult(
            method_name="OpenEMS_FDTD",
            accuracy=1.0,  # Ground truth이므로 100% 정확
            computation_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            convergence_rate=fdtd_results.get('convergence_iterations', 0),
            frequency_range=(scenario.get('frequency', 0), 
                           scenario.get('frequency', 0)),
            scenario=scenario['name']
        ), fdtd_results
    
    def run_emnerf_benchmark(self, scenario, ground_truth):
        """EM-NeRF 방법 벤치마크"""
        print(f"🧠 EM-NeRF 추론: {scenario['name']}")
        
        if self.em_nerf is None:
            print("❌ EM-NeRF 모델이 로드되지 않았습니다")
            return None
        
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        # EM-NeRF로 예측
        nerf_results = self.em_nerf.predict_scenario(scenario)
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        # 정확도 계산 (Ground Truth와 비교)
        accuracy = self.calculate_accuracy(nerf_results, ground_truth)
        
        return BenchmarkResult(
            method_name="EM-NeRF",
            accuracy=accuracy,
            computation_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            convergence_rate=1,  # 추론은 1회로 완료
            frequency_range=(scenario.get('frequency', 0),
                           scenario.get('frequency', 0)),
            scenario=scenario['name']
        )
    
    def run_mom_benchmark(self, scenario, ground_truth):
        """MoM 방법 벤치마크"""
        print(f"⚡ MoM 해석: {scenario['name']}")
        
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        # MoM 해석 실행
        mom_results = self.mom_solver.solve_scenario(scenario)
        
        end_time = time.time() 
        end_memory = self.get_memory_usage()
        
        # 정확도 계산
        accuracy = self.calculate_accuracy(mom_results, ground_truth)
        
        return BenchmarkResult(
            method_name="Method_of_Moments",
            accuracy=accuracy,
            computation_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            convergence_rate=mom_results.get('iterations', 0),
            frequency_range=(scenario.get('frequency', 0),
                           scenario.get('frequency', 0)),
            scenario=scenario['name']
        )
    
    def calculate_accuracy(self, predicted, ground_truth):
        """예측 결과와 Ground Truth 비교"""
        
        # 전기장 비교
        E_pred = predicted['E_field']
        E_true = ground_truth['E_field']
        E_mse = np.mean((E_pred - E_true)**2)
        
        # 자기장 비교  
        H_pred = predicted['H_field']
        H_true = ground_truth['H_field']
        H_mse = np.mean((H_pred - H_true)**2)
        
        # 종합 정확도 (MSE 기반, 0-1 스케일)
        total_mse = (E_mse + H_mse) / 2
        accuracy = np.exp(-total_mse)  # MSE가 낮을수록 정확도 높음
        
        return float(accuracy)
    
    def run_comprehensive_benchmark(self):
        """종합 벤치마크 실행"""
        
        print("🚀 종합 전자기 해석 방법 벤치마크 시작!")
        print("="*60)
        
        scenarios = self.setup_test_scenarios()
        
        for scenario in scenarios:
            print(f"\n📋 시나리오: {scenario['name']}")
            print(f"   복잡도: {scenario['complexity']}")
            print(f"   예상 우세 방법: {scenario['expected_advantage']}")
            print("-" * 40)
            
            # 1. FDTD로 Ground Truth 생성
            fdtd_result, ground_truth = self.run_fdtd_benchmark(scenario)
            self.results.append(fdtd_result)
            
            # 2. EM-NeRF 평가
            nerf_result = self.run_emnerf_benchmark(scenario, ground_truth)
            if nerf_result:
                self.results.append(nerf_result)
            
            # 3. MoM 평가
            mom_result = self.run_mom_benchmark(scenario, ground_truth)
            if mom_result:
                self.results.append(mom_result)
            
            # 시나리오별 결과 출력
            self.print_scenario_results(scenario['name'])
        
        # 종합 분석
        self.analyze_overall_performance()
        self.plot_comprehensive_results()
    
    def print_scenario_results(self, scenario_name):
        """시나리오별 결과 출력"""
        scenario_results = [r for r in self.results if r.scenario == scenario_name]
        
        print(f"\n📊 {scenario_name} 결과:")
        print("=" * 50)
        
        for result in scenario_results:
            print(f"🔸 {result.method_name:15s}")
            print(f"   정확도:    {result.accuracy:.4f}")
            print(f"   계산시간:  {result.computation_time:.3f}초")
            print(f"   메모리:    {result.memory_usage:.1f}MB")
            print(f"   수렴률:    {result.convergence_rate}")
        
        # 최고 성능 방법 찾기
        if len(scenario_results) > 1:
            best_accuracy = max(scenario_results, key=lambda x: x.accuracy)
            best_speed = min(scenario_results, key=lambda x: x.computation_time)
            
            print(f"\n🏆 최고 정확도: {best_accuracy.method_name} ({best_accuracy.accuracy:.4f})")
            print(f"⚡ 최고 속도:   {best_speed.method_name} ({best_speed.computation_time:.3f}초)")
    
    def analyze_overall_performance(self):
        """전체 성능 분석"""
        print("\n" + "="*60)
        print("📈 종합 성능 분석")
        print("="*60)
        
        methods = list(set([r.method_name for r in self.results]))
        
        for method in methods:
            method_results = [r for r in self.results if r.method_name == method]
            
            avg_accuracy = np.mean([r.accuracy for r in method_results])
            avg_time = np.mean([r.computation_time for r in method_results])
            avg_memory = np.mean([r.memory_usage for r in method_results])
            
            print(f"\n🔹 {method}")
            print(f"   평균 정확도: {avg_accuracy:.4f}")
            print(f"   평균 시간:   {avg_time:.3f}초")
            print(f"   평균 메모리: {avg_memory:.1f}MB")
            
            # 적용 분야 추천
            if method == "EM-NeRF":
                if avg_accuracy > 0.9 and avg_time < 1.0:
                    print("   ✅ 추천: 실시간 예측, 복잡한 환경")
                elif avg_accuracy > 0.8:
                    print("   ✅ 추천: 빠른 근사 해석")
                else:
                    print("   ⚠️  개선 필요: 더 많은 훈련 데이터 필요")
            
            elif method == "OpenEMS_FDTD":
                print("   ✅ 추천: 정확한 해석, 광대역 분석")
                if avg_time > 60:
                    print("   ⚠️  단점: 계산 시간이 김")
            
            elif method == "Method_of_Moments":
                if avg_time < avg_time:  # FDTD와 비교
                    print("   ✅ 추천: 중간 복잡도 문제")
                else:
                    print("   ⚠️  제한: 복잡한 구조에는 부적합")
    
    def plot_comprehensive_results(self):
        """결과 시각화"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        methods = list(set([r.method_name for r in self.results]))
        scenarios = list(set([r.scenario for r in self.results]))
        
        # 1. 정확도 비교
        accuracy_data = {}
        for method in methods:
            accuracy_data[method] = [
                np.mean([r.accuracy for r in self.results 
                        if r.method_name == method and r.scenario == scenario])
                for scenario in scenarios
            ]
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        for i, (method, accuracies) in enumerate(accuracy_data.items()):
            ax1.bar(x + i*width, accuracies, width, label=method)
        
        ax1.set_xlabel('시나리오')
        ax1.set_ylabel('정확도')
        ax1.set_title('시나리오별 정확도 비교')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(scenarios, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 계산 시간 비교 (로그 스케일)
        time_data = {}
        for method in methods:
            time_data[method] = [
                np.mean([r.computation_time for r in self.results 
                        if r.method_name == method and r.scenario == scenario])
                for scenario in scenarios
            ]
        
        for i, (method, times) in enumerate(time_data.items()):
            ax2.bar(x + i*width, times, width, label=method)
        
        ax2.set_xlabel('시나리오')
        ax2.set_ylabel('계산 시간 (초)')
        ax2.set_title('시나리오별 계산 시간 비교')
        ax2.set_yscale('log')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(scenarios, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 정확도 vs 속도 산점도
        for method in methods:
            method_results = [r for r in self.results if r.method_name == method]
            accuracies = [r.accuracy for r in method_results]
            times = [r.computation_time for r in method_results]
            
            ax3.scatter(times, accuracies, label=method, s=100, alpha=0.7)
        
        ax3.set_xlabel('계산 시간 (초)')
        ax3.set_ylabel('정확도')
        ax3.set_title('정확도 vs 계산 시간')
        ax3.set_xscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 메모리 사용량 비교
        memory_data = {}
        for method in methods:
            memory_data[method] = [
                np.mean([r.memory_usage for r in self.results 
                        if r.method_name == method and r.scenario == scenario])
                for scenario in scenarios
            ]
        
        for i, (method, memories) in enumerate(memory_data.items()):
            ax4.bar(x + i*width, memories, width, label=method)
        
        ax4.set_xlabel('시나리오')
        ax4.set_ylabel('메모리 사용량 (MB)')
        ax4.set_title('시나리오별 메모리 사용량')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(scenarios, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comprehensive_em_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_memory_usage(self):
        """현재 메모리 사용량 측정"""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB

# 실행 예제
if __name__ == "__main__":
    
    # 1. EM-NeRF 모델 로드 (이미 훈련된 모델)
    emnerf_model = load_trained_emnerf('best_openems_emnerf.pth')
    
    # 2. 벤치마크 시스템 초기화
    benchmark = ComprehensiveEMBenchmark()
    benchmark.em_nerf = emnerf_model
    
    # 3. 종합 벤치마크 실행
    benchmark.run_comprehensive_benchmark()
    
    print("\n🎉 벤치마크 완료!")
    print("📋 결과 파일: comprehensive_em_benchmark.png")
    print("💡 결론: 각 방법의 장단점과 적용 분야를 명확히 파악!")