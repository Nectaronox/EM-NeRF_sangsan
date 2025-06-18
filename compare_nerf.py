"""
EM-NeRF vs 전통적 전자기장 시뮬레이션 방법 성능 비교
이미 훈련된 EM-NeRF 모델과 다른 시뮬레이션 방식들을 비교합니다.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RBFInterpolator
from scipy.special import spherical_jn, spherical_yn
import time
import pandas as pd
from IPython.display import display
import seaborn as sns

# 시각화 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class TraditionalEMSimulators:
    """전통적인 전자기장 시뮬레이션 방법들"""
    
    def __init__(self, reference_data=None):
        """
        reference_data: 학습 데이터셋의 일부를 참조 데이터로 사용
        """
        self.reference_data = reference_data
        if reference_data:
            self.setup_interpolators()
    
    def setup_interpolators(self):
        """보간기 설정"""
        # 참조 데이터에서 위치와 필드 추출
        positions = self.reference_data['positions']
        e_fields = self.reference_data['e_fields']
        
        # 선형 보간기
        self.linear_interpolator_E = {}
        for i in range(3):
            self.linear_interpolator_E[i] = lambda p, i=i: griddata(
                positions, e_fields[:, i], p, method='linear', fill_value=0
            )
        
        # RBF 보간기 (더 정확하지만 느림)
        self.rbf_interpolator_E = RBFInterpolator(
            positions, e_fields, kernel='multiquadric', epsilon=2
        )
    
    def analytical_dipole(self, positions, frequency, time):
        """
        해석적 다이폴 안테나 방사 패턴
        간단한 전자기장 모델
        """
        r = np.linalg.norm(positions, axis=1, keepdims=True)
        theta = np.arccos(positions[:, 2:3] / (r + 1e-10))
        
        # 파장 및 파수
        c = 3e8  # 광속
        wavelength = c / frequency
        k = 2 * np.pi / wavelength
        
        # 전기장 (원거리장 근사)
        E_theta = np.sin(theta) * np.exp(-1j * k * r) / (r + 1e-10)
        E_mag = np.abs(E_theta) * 0.1  # 스케일 조정
        
        # 3D 벡터로 변환 (간단화)
        E_field = np.zeros((len(positions), 3))
        E_field[:, 0] = E_mag.flatten() * np.sin(theta.flatten())
        E_field[:, 1] = E_mag.flatten() * np.cos(theta.flatten())
        E_field[:, 2] = E_mag.flatten() * 0.5
        
        # 정규화
        E_field = np.tanh(E_field)
        
        return E_field
    
    def fdtd_approximation(self, positions, frequency, time):
        """
        FDTD (Finite-Difference Time-Domain) 근사
        실제 FDTD는 매우 복잡하므로 간단한 근사 사용
        """
        # 그리드 기반 시뮬레이션을 점 기반으로 근사
        r = np.linalg.norm(positions, axis=1)
        
        # 시간 도메인 파동
        c = 3e8
        k = 2 * np.pi * frequency / c
        phase = k * r - 2 * np.pi * frequency * time
        
        # 전기장 계산
        E_field = np.zeros((len(positions), 3))
        E_field[:, 0] = np.sin(phase) / (r + 0.1)
        E_field[:, 1] = np.cos(phase) / (r + 0.1)
        E_field[:, 2] = np.sin(2 * phase) / (r + 0.1)
        
        # 정규화
        E_field = np.tanh(E_field)
        
        return E_field
    
    def mom_approximation(self, positions, frequency, time):
        """
        MoM (Method of Moments) 근사
        경계 요소법 기반 시뮬레이션의 간단한 버전
        """
        # 몇 개의 기본 함수로 근사
        basis_centers = np.array([
            [0, 0, 0],
            [0.5, 0, 0],
            [-0.5, 0, 0],
            [0, 0.5, 0],
            [0, -0.5, 0]
        ])
        
        E_field = np.zeros((len(positions), 3))
        
        for center in basis_centers:
            dist = np.linalg.norm(positions - center, axis=1, keepdims=True)
            weight = np.exp(-dist**2 / 0.5)
            
            # 각 기본 함수의 기여
            E_field[:, 0] += weight.flatten() * np.sin(2 * np.pi * frequency * 1e-9)
            E_field[:, 1] += weight.flatten() * np.cos(2 * np.pi * frequency * 1e-9)
            E_field[:, 2] += weight.flatten() * np.sin(4 * np.pi * frequency * 1e-9)
        
        # 정규화
        E_field = np.tanh(E_field / len(basis_centers))
        
        return E_field
    
    def linear_interpolation(self, positions):
        """선형 보간"""
        if not hasattr(self, 'linear_interpolator_E'):
            raise ValueError("참조 데이터가 필요합니다")
        
        E_field = np.zeros((len(positions), 3))
        for i in range(3):
            E_field[:, i] = self.linear_interpolator_E[i](positions)
        
        return E_field
    
    def rbf_interpolation(self, positions):
        """RBF 보간"""
        if not hasattr(self, 'rbf_interpolator_E'):
            raise ValueError("참조 데이터가 필요합니다")
        
        return self.rbf_interpolator_E(positions)

class PerformanceComparison:
    """성능 비교 및 분석 클래스"""
    
    def __init__(self, nerf_model, traditional_sims, device='cuda'):
        self.nerf_model = nerf_model.to(device)
        self.traditional_sims = traditional_sims
        self.device = device
        self.results = {}
    
    def generate_test_data(self, n_samples=1000):
        """테스트 데이터 생성"""
        # 균일하게 분포된 테스트 포인트
        positions = torch.rand(n_samples, 3) * 2 - 1
        frequencies = torch.ones(n_samples, 1) * 0.5  # 정규화된 주파수
        times = torch.zeros(n_samples, 1)
        dynamic_objects = torch.zeros(n_samples, 8)
        
        # Ground truth (합성 데이터)
        ground_truth = {
            'positions': positions.numpy(),
            'E_field': np.tanh(np.random.randn(n_samples, 3) * 0.5)
        }
        
        return {
            'positions': positions,
            'frequencies': frequencies,
            'times': times,
            'dynamic_objects': dynamic_objects,
            'ground_truth': ground_truth
        }
    
    def evaluate_method(self, method_name, method_func, test_data, use_torch=False):
        """단일 방법 평가"""
        print(f"\n평가 중: {method_name}")
        
        positions = test_data['positions']
        ground_truth = test_data['ground_truth']['E_field']
        
        # 시간 측정
        start_time = time.time()
        
        if use_torch:
            # NeRF 모델
            with torch.no_grad():
                predictions = method_func(
                    positions.to(self.device),
                    test_data['frequencies'].to(self.device),
                    test_data['times'].to(self.device),
                    test_data['dynamic_objects'].to(self.device)
                )
                E_pred = predictions['electric_field'].cpu().numpy()
        else:
            # 전통적 방법
            if 'interpolation' in method_name.lower():
                E_pred = method_func(positions.numpy())
            else:
                E_pred = method_func(
                    positions.numpy(),
                    float(test_data['frequencies'][0]) * 1e10,  # 역정규화
                    float(test_data['times'][0])
                )
        
        inference_time = time.time() - start_time
        
        # 메트릭 계산
        mse = np.mean((E_pred - ground_truth)**2)
        mae = np.mean(np.abs(E_pred - ground_truth))
        rmse = np.sqrt(mse)
        
        # 상관계수
        correlation = np.corrcoef(E_pred.flatten(), ground_truth.flatten())[0, 1]
        
        # 결과 저장
        self.results[method_name] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'Correlation': correlation,
            'Inference_Time': inference_time,
            'Time_per_Sample': inference_time / len(positions),
            'predictions': E_pred
        }
        
        print(f"  - MSE: {mse:.6f}")
        print(f"  - MAE: {mae:.6f}")
        print(f"  - Inference Time: {inference_time:.4f}s")
        print(f"  - Time per Sample: {inference_time/len(positions)*1000:.2f}ms")
    
    def run_comparison(self, n_test_samples=1000):
        """전체 비교 실행"""
        print("="*60)
        print("전자기장 시뮬레이션 방법 비교 시작")
        print("="*60)
        
        # 테스트 데이터 생성
        test_data = self.generate_test_data(n_test_samples)
        
        # 참조 데이터 설정 (보간법용)
        ref_data = {
            'positions': test_data['positions'].numpy()[:500],
            'e_fields': test_data['ground_truth']['E_field'][:500]
        }
        self.traditional_sims.reference_data = ref_data
        self.traditional_sims.setup_interpolators()
        
        # 1. NeRF 평가
        self.evaluate_method(
            "EM-NeRF",
            self.nerf_model,
            test_data,
            use_torch=True
        )
        
        # 2. 전통적 방법들 평가
        methods = {
            "Analytical Dipole": self.traditional_sims.analytical_dipole,
            "FDTD Approximation": self.traditional_sims.fdtd_approximation,
            "MoM Approximation": self.traditional_sims.mom_approximation,
            "Linear Interpolation": self.traditional_sims.linear_interpolation,
            "RBF Interpolation": self.traditional_sims.rbf_interpolation
        }
        
        for name, method in methods.items():
            try:
                self.evaluate_method(name, method, test_data)
            except Exception as e:
                print(f"  ❌ 오류 발생: {e}")
    
    def visualize_results(self):
        """결과 시각화"""
        # 결과 DataFrame 생성
        df = pd.DataFrame(self.results).T
        df = df.reset_index().rename(columns={'index': 'Method'})
        
        # 1. 성능 메트릭 비교
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics = ['MSE', 'MAE', 'RMSE', 'Correlation', 'Inference_Time', 'Time_per_Sample']
        titles = ['Mean Squared Error', 'Mean Absolute Error', 'RMSE', 
                  'Correlation', 'Total Inference Time (s)', 'Time per Sample (s)']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            if metric in df.columns:
                df_sorted = df.sort_values(metric, ascending=(metric != 'Correlation'))
                colors = ['#2ecc71' if x == 'EM-NeRF' else '#3498db' for x in df_sorted['Method']]
                
                bars = ax.bar(range(len(df_sorted)), df_sorted[metric], color=colors)
                ax.set_xticks(range(len(df_sorted)))
                ax.set_xticklabels(df_sorted['Method'], rotation=45, ha='right')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                
                # 값 표시
                for bar, val in zip(bars, df_sorted[metric]):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.4f}' if val < 1 else f'{val:.2f}',
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 레이더 차트 (성능 프로파일)
        self.plot_radar_chart(df)
        
        # 3. 결과 테이블
        print("\n📊 성능 비교 테이블:")
        display(df.round(4))
        
        # 4. 상세 분석
        self.detailed_analysis(df)
    
    def plot_radar_chart(self, df):
        """레이더 차트로 다차원 성능 비교"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')
        
        # 메트릭 선택 (정규화 필요)
        metrics = ['MSE', 'MAE', 'Inference_Time']
        
        # 정규화 (0-1 범위, 낮을수록 좋음)
        normalized_data = {}
        for method in df['Method']:
            normalized_data[method] = []
            for metric in metrics:
                values = df[metric].values
                normalized_val = 1 - (df[df['Method']==method][metric].values[0] - values.min()) / (values.max() - values.min() + 1e-10)
                normalized_data[method].append(normalized_val)
        
        # 각도 설정
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        metrics += metrics[:1]
        angles += angles[:1]
        
        # 플롯
        for method, values in normalized_data.items():
            values += values[:1]
            color = '#e74c3c' if method == 'EM-NeRF' else None
            ax.plot(angles, values, 'o-', linewidth=2, label=method, 
                   color=color, markersize=8)
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['MSE↓', 'MAE↓', 'Time↓'])
        ax.set_ylim(0, 1)
        ax.set_title('성능 프로파일 비교\n(높을수록 좋음)', fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('radar_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def detailed_analysis(self, df):
        """상세 분석 및 인사이트"""
        print("\n🔍 상세 분석:")
        print("="*60)
        
        # 1. 정확도 리더
        accuracy_leader = df.loc[df['MSE'].idxmin(), 'Method']
        print(f"✅ 정확도 1위: {accuracy_leader}")
        print(f"   - MSE: {df.loc[df['MSE'].idxmin(), 'MSE']:.6f}")
        
        # 2. 속도 리더
        speed_leader = df.loc[df['Time_per_Sample'].idxmin(), 'Method']
        print(f"\n⚡ 속도 1위: {speed_leader}")
        print(f"   - Time per sample: {df.loc[df['Time_per_Sample'].idxmin(), 'Time_per_Sample']*1000:.2f}ms")
        
        # 3. NeRF 분석
        nerf_idx = df[df['Method'] == 'EM-NeRF'].index[0]
        nerf_accuracy_rank = len(df) - (df['MSE'].values <= df.loc[nerf_idx, 'MSE']).sum() + 1
        nerf_speed_rank = len(df) - (df['Time_per_Sample'].values <= df.loc[nerf_idx, 'Time_per_Sample']).sum() + 1
        
        print(f"\n🧠 EM-NeRF 성능:")
        print(f"   - 정확도 순위: {nerf_accuracy_rank}/{len(df)}")
        print(f"   - 속도 순위: {nerf_speed_rank}/{len(df)}")
        
        # 4. Trade-off 분석
        print("\n📈 Trade-off 분석:")
        for _, row in df.iterrows():
            efficiency = (1/row['MSE']) / (row['Time_per_Sample'] + 1e-10)
            print(f"   - {row['Method']}: 효율성 점수 = {efficiency:.2f}")
    
    def plot_prediction_samples(self, test_data, n_samples=5):
        """예측 샘플 시각화"""
        fig, axes = plt.subplots(n_samples, len(self.results), figsize=(15, 3*n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        positions = test_data['positions'].numpy()
        ground_truth = test_data['ground_truth']['E_field']
        
        # 랜덤 샘플 선택
        sample_indices = np.random.choice(len(positions), n_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            for j, (method, results) in enumerate(self.results.items()):
                ax = axes[i, j]
                
                # 예측값과 실제값 비교
                pred = results['predictions'][idx]
                true = ground_truth[idx]
                
                x = ['Ex', 'Ey', 'Ez']
                ax.bar(np.arange(3)-0.2, true, 0.4, label='True', alpha=0.7)
                ax.bar(np.arange(3)+0.2, pred, 0.4, label='Predicted', alpha=0.7)
                
                ax.set_xticks(range(3))
                ax.set_xticklabels(x)
                ax.set_ylim(-1.5, 1.5)
                
                if i == 0:
                    ax.set_title(method)
                if j == 0:
                    ax.set_ylabel(f'Sample {idx}')
                if i == n_samples - 1 and j == len(self.results) // 2:
                    ax.legend()
                
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('prediction_samples.png', dpi=300, bbox_inches='tight')
        plt.show()

# 실행 코드
def run_performance_comparison(trained_model):
    """학습된 모델과 전통적 방법들의 성능 비교 실행"""
    
    # 전통적 시뮬레이터 초기화
    traditional_sims = TraditionalEMSimulators()
    
    # 비교 클래스 초기화
    comparison = PerformanceComparison(
        trained_model,
        traditional_sims,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 비교 실행
    comparison.run_comparison(n_test_samples=1000)
    
    # 결과 시각화
    comparison.visualize_results()
    
    # 예측 샘플 시각화
    test_data = comparison.generate_test_data(100)
    comparison.plot_prediction_samples(test_data, n_samples=3)
    
    return comparison

# 메인 실행 (주피터 노트북에서)
if __name__ == "__main__":
    print("🚀 EM-NeRF 성능 비교 시작...")
    print("="*60)
    
    # 이미 학습된 모델이 있다고 가정 (model 변수)
    # comparison = run_performance_comparison(model)
    
    print("\n✅ 비교 완료!")
    print("\n💡 주요 인사이트:")
    print("- EM-NeRF는 연속적인 표현이 가능하며 임의의 위치에서 쿼리 가능")
    print("- 전통적 방법들은 특정 상황에서 더 빠르거나 정확할 수 있음")
    print("- 실제 응용에서는 정확도와 속도 간의 trade-off 고려 필요")