#!/usr/bin/env python3
"""
RunPod GPU 테스트용 전자기장 NeRF + 동작분석 시스템
실제 성능 측정 및 벤치마크를 위한 완전한 구현

실행 방법:
1. RunPod 인스턴스에 이 파일 업로드
2. python runpod_em_nerf_test.py
3. 결과 확인

필요한 GPU: RTX 4090, A100, H100 등 (최소 12GB VRAM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from datetime import datetime
import psutil
import GPUtil
from typing import Dict, List, Tuple
import h5py
import cv2
from tqdm import tqdm
import logging

# GPU 메모리 모니터링 설정
torch.cuda.empty_cache()
if torch.cuda.is_available():
    print(f"CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntheticEMDataset(Dataset):
    """합성 전자기장 데이터셋 생성"""
    
    def __init__(self, n_samples=50000, grid_size=64, frequency_range=(1e9, 10e9)):
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.frequency_range = frequency_range
        
        logger.info(f"데이터셋 생성 중: {n_samples} 샘플, 그리드 크기: {grid_size}")
        
        
        self.positions = torch.rand(n_samples, 3) * 2 - 1 
            
        self.frequencies = torch.rand(n_samples, 1) * (frequency_range[1] - frequency_range[0]) + frequency_range[0]

        self.times = torch.rand(n_samples, 1) * 1e-6  
        
        self.dynamic_objects = self.generate_dynamic_objects(n_samples)
        
        self.em_fields = self.simulate_ground_truth_fields()
        
        logger.info("데이터셋 생성 완료")
    
    def generate_dynamic_objects(self, n_samples):
        """동적 객체 정보 생성"""
        # [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, object_type, size]
        objects = torch.zeros(n_samples, 8)
        
        # 위치 ([-1, 1] 범위)
        objects[:, :3] = torch.rand(n_samples, 3) * 2 - 1
        
        # 속도 ([-10, 10] m/s 범위)
        objects[:, 3:6] = torch.rand(n_samples, 3) * 20 - 10
        
        # 객체 타입 (0: person, 1: car, 2: drone, 3: bird, 4: other)
        objects[:, 6] = torch.randint(0, 5, (n_samples,)).float()
        
        # 크기 (0.1 ~ 5.0 미터)
        objects[:, 7] = torch.rand(n_samples) * 4.9 + 0.1
        
        return objects
    
    def simulate_ground_truth_fields(self):
        """물리 기반 ground truth 전자기장 시뮬레이션"""
        logger.info("Ground truth 전자기장 계산 중...")
        
        # Maxwell 방정식 기반 간단한 시뮬레이션
        n_samples = len(self.positions)
        
        # 전기장 (3차원)
        E_field = torch.zeros(n_samples, 3)
        
        # 자기장 (3차원)
        B_field = torch.zeros(n_samples, 3)
        
        # 산란 계수
        scattering = torch.zeros(n_samples, 1)
        
        # 전파 지연
        delay = torch.zeros(n_samples, 1)
        
        for i in range(n_samples):
            pos = self.positions[i]
            freq = self.frequencies[i].item()
            obj = self.dynamic_objects[i]
            
            # 거리 기반 감쇠
            distance = torch.norm(pos)
            
            # 주파수 의존 전기장
            wavelength = 3e8 / freq
            k = 2 * np.pi / wavelength
            
            # 간단한 쌍극자 복사 모델
            E_field[i, 0] = torch.sin(k * distance) / (distance + 0.1)
            E_field[i, 1] = torch.cos(k * distance) / (distance + 0.1)
            E_field[i, 2] = torch.sin(2 * k * distance) / (distance + 0.1)
            
            # 자기장 (E와 수직)
            B_field[i] = torch.cross(E_field[i], torch.tensor([0., 0., 1.]))
            B_field[i] /= 3e8  # c로 나누어 자기장 크기 조정
            
            # 객체에 의한 산란 (크기와 속도에 의존)
            obj_influence = torch.norm(obj[:3] - pos) 
            scattering[i] = torch.sigmoid(-obj_influence + obj[7] * 0.1)  # 크기 의존
            
            # 전파 지연 (거리와 매질에 의존)
            delay[i] = distance / 3e8 + scattering[i] * 1e-9
        
        return {
            'electric_field': E_field,
            'magnetic_field': B_field,
            'scattering_coefficient': scattering,
            'propagation_delay': delay
        }
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'position': self.positions[idx],
            'frequency': self.frequencies[idx],
            'time': self.times[idx],
            'dynamic_objects': self.dynamic_objects[idx],
            'ground_truth': {k: v[idx] for k, v in self.em_fields.items()}
        }

class OptimizedEMNeRF(nn.Module):
    """RunPod 최적화된 전자기장 NeRF"""
    
    def __init__(self, hidden_dim=256, n_layers=8, use_mixed_precision=True):
        super().__init__()
        
        self.use_mixed_precision = use_mixed_precision
        
        # 위치 인코딩 (효율적인 구현)
        self.pos_encoding_levels = 10
        self.time_encoding_levels = 4
        
        # 입력 차원 계산
        pos_dim = 3 + 3 * 2 * self.pos_encoding_levels  # 원본 + sin/cos 인코딩
        time_dim = 1 + 1 * 2 * self.time_encoding_levels
        freq_dim = 8  # 주파수 인코딩
        obj_dim = 8   # 동적 객체
        
        input_dim = pos_dim + time_dim + freq_dim + obj_dim
        
        # 효율적인 backbone 네트워크
        layers = []
        current_dim = input_dim
        
        for i in range(n_layers):
            # Skip connection at halfway point
            if i == n_layers // 2:
                layers.append(nn.Linear(current_dim + input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))
                
            layers.append(nn.LayerNorm(hidden_dim))  # 안정성을 위한 LayerNorm
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.1))  # 오버피팅 방지
            current_dim = hidden_dim
        
        self.backbone = nn.ModuleList(layers)
        
        # 출력 헤드들 (더 효율적으로)
        head_dim = hidden_dim // 2
        
        self.electric_field_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim, 3)
        )
        
        self.magnetic_field_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim, 3)
        )
        
        self.scattering_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.delay_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim // 2, 1)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def positional_encoding(self, x, levels):
        """효율적인 위치 인코딩"""
        encoded = [x]
        for level in range(levels):
            for fn in [torch.sin, torch.cos]:
                encoded.append(fn(2**level * np.pi * x))
        return torch.cat(encoded, dim=-1)
    
    def frequency_encoding(self, freq):
        """주파수 인코딩"""
        # 로그 스케일로 정규화
        log_freq = torch.log10(torch.clamp(freq, min=1e6, max=1e12))
        normalized = (log_freq - 6) / 6  # [1MHz, 1THz] -> [-1, 1]
        
        # 위치 인코딩 스타일
        encoded = [normalized]
        for i in range(3):
            encoded.append(torch.sin(2**i * np.pi * normalized))
            encoded.append(torch.cos(2**i * np.pi * normalized))
        
        return torch.cat(encoded, dim=-1)
    
    def forward(self, positions, frequencies, times, dynamic_objects):
        batch_size = positions.shape[0]
        
        # 인코딩
        pos_encoded = self.positional_encoding(positions, self.pos_encoding_levels)
        time_encoded = self.positional_encoding(times, self.time_encoding_levels)
        freq_encoded = self.frequency_encoding(frequencies)
        
        # 동적 객체 정규화
        obj_normalized = torch.tanh(dynamic_objects)
        
        # 특징 결합
        features = torch.cat([pos_encoded, time_encoded, freq_encoded, obj_normalized], dim=1)
        
        # Backbone 통과 (skip connection 포함)
        skip_input = features
        x = features
        
        for i, layer in enumerate(self.backbone):
            if isinstance(layer, nn.Linear) and layer.in_features == x.shape[1] + skip_input.shape[1]:
                # Skip connection
                x = torch.cat([x, skip_input], dim=1)
            x = layer(x)
        
        # 출력 계산
        E_field = self.electric_field_head(x)
        B_field = self.magnetic_field_head(x)
        scattering = self.scattering_head(x)
        delay = self.delay_head(x)
        
        return {
            'electric_field': E_field,
            'magnetic_field': B_field,
            'scattering_coefficient': scattering,
            'propagation_delay': delay
        }

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        self.metrics = {
            'gpu_memory': [],
            'gpu_utilization': [],
            'training_loss': [],
            'inference_time': [],
            'throughput': [],
            'cpu_usage': [],
            'system_memory': []
        }
        
    def update(self, **kwargs):
        # GPU 정보 수집
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            self.metrics['gpu_memory'].append(gpu.memoryUsed / gpu.memoryTotal * 100)
            self.metrics['gpu_utilization'].append(gpu.load * 100)
        
        # CPU 및 시스템 메모리
        self.metrics['cpu_usage'].append(psutil.cpu_percent())
        self.metrics['system_memory'].append(psutil.virtual_memory().percent)
        
        # 추가 메트릭
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_summary(self):
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'std': np.std(values)
                }
        return summary
    
    def save_report(self, filepath):
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
                'cpu_count': os.cpu_count(),
                'system_memory': psutil.virtual_memory().total / 1e9
            },
            'metrics': self.get_summary(),
            'raw_data': self.metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"성능 리포트 저장됨: {filepath}")

class EMNeRFTrainer:
    """최적화된 NeRF 트레이너"""
    
    def __init__(self, model, device='cuda', use_mixed_precision=True):
        self.model = model.to(device)
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        
        # 옵티마이저 (AdamW 사용)
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        
        # 스케줄러
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
        # 성능 모니터
        self.monitor = PerformanceMonitor()
    
    def physics_loss(self, predictions, targets):
        """물리 기반 손실 함수"""
        mse_loss = nn.MSELoss()
        
        # 기본 MSE 손실
        e_loss = mse_loss(predictions['electric_field'], targets['electric_field'])
        b_loss = mse_loss(predictions['magnetic_field'], targets['magnetic_field'])
        s_loss = mse_loss(predictions['scattering_coefficient'], targets['scattering_coefficient'])
        d_loss = mse_loss(predictions['propagation_delay'], targets['propagation_delay'])
        
        # 물리 법칙 제약 (간단한 버전)
        # |E| 와 |B| 의 관계: |B| ≈ |E|/c
        E_magnitude = torch.norm(predictions['electric_field'], dim=1)
        B_magnitude = torch.norm(predictions['magnetic_field'], dim=1)
        c = 3e8
        physics_constraint = mse_loss(B_magnitude * c, E_magnitude)
        
        total_loss = e_loss + b_loss + s_loss + d_loss + 0.1 * physics_constraint
        
        return {
            'total': total_loss,
            'electric': e_loss,
            'magnetic': b_loss,
            'scattering': s_loss,
            'delay': d_loss,
            'physics': physics_constraint
        }
    
    def train_epoch(self, dataloader):
        self.model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            # 데이터 GPU로 이동
            positions = batch['position'].to(self.device)
            frequencies = batch['frequency'].to(self.device)
            times = batch['time'].to(self.device)
            dynamic_objects = batch['dynamic_objects'].to(self.device)
            
            targets = {k: v.to(self.device) for k, v in batch['ground_truth'].items()}
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    predictions = self.model(positions, frequencies, times, dynamic_objects)
                    losses = self.physics_loss(predictions, targets)
                    total_loss = losses['total']
                
                # Backward pass
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(positions, frequencies, times, dynamic_objects)
                losses = self.physics_loss(predictions, targets)
                total_loss = losses['total']
                
                total_loss.backward()
                self.optimizer.step()
            
            epoch_losses.append(total_loss.item())
            
            # 성능 모니터링
            if batch_idx % 10 == 0:
                self.monitor.update(training_loss=total_loss.item())
        
        self.scheduler.step()
        return np.mean(epoch_losses)
    
    def benchmark_inference(self, test_dataloader, n_iterations=100):
        """추론 성능 벤치마크"""
        self.model.eval()
        inference_times = []
        
        logger.info(f"추론 벤치마크 시작 ({n_iterations} 반복)")
        
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                if i >= n_iterations:
                    break
                
                # 데이터 준비
                positions = batch['position'].to(self.device)
                frequencies = batch['frequency'].to(self.device)
                times = batch['time'].to(self.device)
                dynamic_objects = batch['dynamic_objects'].to(self.device)
                
                # 추론 시간 측정
                torch.cuda.synchronize()
                start_time = time.time()
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(positions, frequencies, times, dynamic_objects)
                else:
                    predictions = self.model(positions, frequencies, times, dynamic_objects)
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                inference_time = end_time - start_time
                inference_times.append(inference_time)
                
                # 처리량 계산 (samples per second)
                throughput = len(positions) / inference_time
                self.monitor.update(inference_time=inference_time, throughput=throughput)
        
        return {
            'mean_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'max_inference_time': np.max(inference_times),
            'min_inference_time': np.min(inference_times),
            'mean_throughput': np.mean([len(batch['position']) / t for t in inference_times])
        }

def run_full_benchmark():
    """전체 벤치마크 실행"""
    logger.info("=== RunPod 전자기장 NeRF 벤치마크 시작 ===")
    
    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"사용 디바이스: {device}")
    
    # 데이터셋 생성
    logger.info("데이터셋 생성 중...")
    train_dataset = SyntheticEMDataset(n_samples=10000, grid_size=64)
    test_dataset = SyntheticEMDataset(n_samples=2000, grid_size=64)
    
    # 데이터로더
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # 모델 생성
    logger.info("모델 초기화 중...")
    model = OptimizedEMNeRF(hidden_dim=256, n_layers=8, use_mixed_precision=True)
    
    # 트레이너 생성
    trainer = EMNeRFTrainer(model, device=device, use_mixed_precision=True)
    
    # 훈련 실행
    logger.info("훈련 시작...")
    n_epochs = 50
    
    for epoch in range(n_epochs):
        start_time = time.time()
        epoch_loss = trainer.train_epoch(train_loader)
        epoch_time = time.time() - start_time
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{n_epochs}, Loss: {epoch_loss:.6f}, Time: {epoch_time:.2f}s")
    
    # 추론 벤치마크
    logger.info("추론 벤치마크 실행 중...")
    inference_results = trainer.benchmark_inference(test_loader, n_iterations=50)
    
    # 결과 출력
    logger.info("\n=== 벤치마크 결과 ===")
    logger.info(f"평균 추론 시간: {inference_results['mean_inference_time']:.4f}s")
    logger.info(f"평균 처리량: {inference_results['mean_throughput']:.1f} samples/sec")
    
    # 성능 리포트 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"em_nerf_benchmark_{timestamp}.json"
    trainer.monitor.save_report(report_path)
    
    # 모델 저장
    model_path = f"em_nerf_model_{timestamp}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'benchmark_results': inference_results,
        'model_config': {
            'hidden_dim': 256,
            'n_layers': 8,
            'use_mixed_precision': True
        }
    }, model_path)
    
    logger.info(f"모델 저장됨: {model_path}")
    
    return {
        'model_path': model_path,
        'report_path': report_path,
        'benchmark_results': inference_results,
        'performance_summary': trainer.monitor.get_summary()
    }

def visualize_results(model_path, save_plots=True):
    """결과 시각화"""
    logger.info("결과 시각화 생성 중...")
    
    # 모델 로드
    checkpoint = torch.load(model_path)
    model = OptimizedEMNeRF()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # 테스트 데이터 생성
    n_points = 1000
    positions = torch.rand(n_points, 3) * 2 - 1
    frequencies = torch.full((n_points, 1), 2.4e9)  # 2.4GHz
    times = torch.zeros(n_points, 1)
    dynamic_objects = torch.zeros(n_points, 8)
    
    # GPU로 이동
    positions = positions.to(device)
    frequencies = frequencies.to(device)
    times = times.to(device)
    dynamic_objects = dynamic_objects.to(device)
    
    # 예측
    with torch.no_grad():
        predictions = model(positions, frequencies, times, dynamic_objects)
    
    # CPU로 이동
    for key in predictions:
        predictions[key] = predictions[key].cpu().numpy()
    positions = positions.cpu().numpy()
    
    # 플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 전기장 크기
    E_magnitude = np.linalg.norm(predictions['electric_field'], axis=1)
    scatter1 = axes[0, 0].scatter(positions[:, 0], positions[:, 1], c=E_magnitude, cmap='viridis')
    axes[0, 0].set_title('Electric Field Magnitude')
    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Y Position')
    plt.colorbar(scatter1, ax=axes[0, 0])
    
    # 자기장 크기
    B_magnitude = np.linalg.norm(predictions['magnetic_field'], axis=1)
    scatter2 = axes[0, 1].scatter(positions[:, 0], positions[:, 1], c=B_magnitude, cmap='plasma')
    axes[0, 1].set_title('Magnetic Field Magnitude')
    axes[0, 1].set_xlabel('X Position')
    axes[0, 1].set_ylabel('Y Position')
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    # 산란 계수
    scattering = predictions['scattering_coefficient'].flatten()
    scatter3 = axes[1, 0].scatter(positions[:, 0], positions[:, 1], c=scattering, cmap='coolwarm')
    axes[1, 0].set_title('Scattering Coefficient')
    axes[1, 0].set_xlabel('X Position')
    axes[1, 0].set_ylabel('Y Position')
    plt.colorbar(scatter3, ax=axes[1, 0])
    
    # 전파 지연
    delay = predictions['propagation_delay'].flatten() * 1e9  # nanoseconds
    scatter4 = axes[1, 1].scatter(positions[:, 0], positions[:, 1], c=delay, cmap='inferno')
    axes[1, 1].set_title('Propagation Delay (ns)')
    axes[1, 1].set_xlabel('X Position')
    axes[1, 1].set_ylabel('Y Position')
    plt.colorbar(scatter4, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"em_field_visualization_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"시각화 저장됨: {plot_path}")
        return plot_path
    else:
        plt.show()
        return None

if __name__ == "__main__":
    try:
        # 전체 벤치마크 실행
        results = run_full_benchmark()
        
        # 결과 시각화
        plot_path = visualize_results(results['model_path'])
        
        print("\n" + "="*50)
        print("🎯 RunPod 벤치마크 완료!")
        print("="*50)
        print(f"📊 성능 리포트: {results['report_path']}")
        print(f"🤖 모델 파일: {results['model_path']}")
        print(f"📈 시각화: {plot_path}")
        print("\n주요 성능 지표:")
        benchmark = results['benchmark_results']
        print(f"  • 평균 추론 시간: {benchmark['mean_inference_time']:.4f}초")
        print(f"  • 처리량: {benchmark['mean_throughput']:.1f} samples/sec")
        print(f"  • GPU 메모리 사용률: {results['performance_summary'].get('gpu_memory', {}).get('mean', 0):.1f}%")
        
        # 실제 데이터셋 다운로드 가이드 출력
        print("\n" + "="*50)
        print("📁 실제 데이터셋 사용 가이드")
        print("="*50)
        print("1. 공개 RF 데이터셋:")
        print("   • NIST RF 전파 측정 데이터: https://its.ntia.gov/")
        print("   • DeepSig RadioML 데이터셋: https://www.deepsig.ai/datasets/")
        print("   • Kaggle RF Signal 데이터: https://www.kaggle.com/datasets/suraj520/rf-signal-data")
        print("\n2. 시뮬레이션 데이터:")
        print("   • MATLAB RF Propagation 툴박스 출력")
        print("   • CST Studio Suite 시뮬레이션 결과")
        print("   • FEKO 전자기장 시뮬레이션")
        
    except Exception as e:
        logger.error(f"벤치마크 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

# 추가 유틸리티 함수들

def download_real_datasets():
    """실제 데이터셋 다운로드 및 전처리 함수"""
    import urllib.request
    import zipfile
    
    logger.info("실제 RF 데이터셋 다운로드 중...")
    
    datasets = {
        'deepsig_radioml': {
            'url': 'https://www.deepsig.ai/datasets/2016.10a.tar.bz2',
            'description': 'RadioML 2016.10a - 변조 인식용 RF 신호 데이터'
        },
        'nist_rf_data': {
            'url': 'https://its.ntia.gov/public-data/spectrum-measurements/',
            'description': 'NIST RF 전파 측정 데이터 (메타데이터만)'
        }
    }
    
    # 실제 다운로드는 용량 문제로 주석 처리
    # 사용자가 필요에 따라 활성화
    """
    for name, info in datasets.items():
        try:
            print(f"다운로드 중: {name}")
            print(f"설명: {info['description']}")
            print(f"URL: {info['url']}")
            # urllib.request.urlretrieve(info['url'], f"{name}.tar.bz2")
            print("수동 다운로드 필요 (용량 제한)")
        except Exception as e:
            print(f"다운로드 실패 {name}: {e}")
    """
    
    return datasets

def create_runpod_setup_script():
    """RunPod 설정 스크립트 생성"""
    setup_script = """#!/bin/bash
# RunPod GPU 인스턴스 설정 스크립트

echo "RunPod 전자기장 NeRF 환경 설정 시작..."

# 필요한 패키지 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib numpy tqdm h5py opencv-python psutil gputil
pip install scipy scikit-learn pandas seaborn

# 메모리 최적화 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 작업 디렉토리 생성
mkdir -p /workspace/em_nerf_test
cd /workspace/em_nerf_test

# 실행 권한 부여
chmod +x runpod_em_nerf_test.py

echo "설정 완료! 다음 명령어로 벤치마크를 시작하세요:"
echo "python runpod_em_nerf_test.py"
"""
    
    with open("setup_runpod.sh", "w") as f:
        f.write(setup_script)
    
    logger.info("RunPod 설정 스크립트 생성됨: setup_runpod.sh")

def memory_profiler():
    """GPU 메모리 사용량 프로파일링"""
    if not torch.cuda.is_available():
        print("CUDA 사용 불가능")
        return
    
    print("\n=== GPU 메모리 프로파일링 ===")
    
    # 다양한 배치 크기에서 메모리 사용량 테스트
    model = OptimizedEMNeRF(hidden_dim=256, n_layers=8).cuda()
    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
    
    results = []
    
    for batch_size in batch_sizes:
        try:
            torch.cuda.empty_cache()
            
            # 테스트 데이터 생성
            positions = torch.rand(batch_size, 3).cuda()
            frequencies = torch.rand(batch_size, 1).cuda() * 1e9 + 1e9
            times = torch.rand(batch_size, 1).cuda() * 1e-6
            dynamic_objects = torch.rand(batch_size, 8).cuda()
            
            # 메모리 측정
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated() / 1e6  # MB
            
            with torch.no_grad():
                output = model(positions, frequencies, times, dynamic_objects)
            
            torch.cuda.synchronize()
            memory_after = torch.cuda.memory_allocated() / 1e6  # MB
            
            memory_used = memory_after - memory_before
            
            results.append({
                'batch_size': batch_size,
                'memory_used_mb': memory_used,
                'memory_per_sample': memory_used / batch_size if batch_size > 0 else 0
            })
            
            print(f"배치 크기 {batch_size:3d}: {memory_used:6.1f} MB ({memory_used/batch_size:.2f} MB/sample)")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"배치 크기 {batch_size}: GPU 메모리 부족")
                break
            else:
                raise e
    
    return results

def create_data_conversion_utils():
    """실제 데이터셋 변환 유틸리티"""
    
    def convert_matlab_to_pytorch(mat_file_path):
        """MATLAB .mat 파일을 PyTorch 텐서로 변환"""
        try:
            import scipy.io
            
            mat_data = scipy.io.loadmat(mat_file_path)
            
            # 일반적인 RF 데이터 구조 가정
            if 'signal_data' in mat_data:
                signals = torch.tensor(mat_data['signal_data'], dtype=torch.float32)
            
            if 'coordinates' in mat_data:
                coordinates = torch.tensor(mat_data['coordinates'], dtype=torch.float32)
            
            if 'frequencies' in mat_data:
                frequencies = torch.tensor(mat_data['frequencies'], dtype=torch.float32)
            
            return {
                'signals': signals,
                'coordinates': coordinates,
                'frequencies': frequencies
            }
            
        except ImportError:
            print("scipy 설치 필요: pip install scipy")
            return None
        except Exception as e:
            print(f"MATLAB 파일 변환 실패: {e}")
            return None
    
    def convert_cst_to_pytorch(cst_export_path):
        """CST Studio Suite 내보내기 파일을 PyTorch 형태로 변환"""
        # CST는 보통 ASCII 형태로 데이터를 내보냄
        try:
            import pandas as pd
            
            # CSV 형태로 내보낸 경우
            if cst_export_path.endswith('.csv'):
                df = pd.read_csv(cst_export_path)
                
                # 일반적인 컬럼명들
                position_cols = ['X', 'Y', 'Z']
                field_cols = ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']
                
                positions = torch.tensor(df[position_cols].values, dtype=torch.float32)
                
                # 전기장
                if all(col in df.columns for col in ['Ex', 'Ey', 'Ez']):
                    e_field = torch.tensor(df[['Ex', 'Ey', 'Ez']].values, dtype=torch.float32)
                else:
                    e_field = None
                
                # 자기장
                if all(col in df.columns for col in ['Bx', 'By', 'Bz']):
                    b_field = torch.tensor(df[['Bx', 'By', 'Bz']].values, dtype=torch.float32)
                else:
                    b_field = None
                
                return {
                    'positions': positions,
                    'electric_field': e_field,
                    'magnetic_field': b_field
                }
            
        except Exception as e:
            print(f"CST 파일 변환 실패: {e}")
            return None
    
    return convert_matlab_to_pytorch, convert_cst_to_pytorch

# 성능 비교를 위한 기준 모델들
class BaselineModels:
    """비교를 위한 기준 모델들"""
    
    @staticmethod
    def simple_mlp():
        """간단한 MLP 기준 모델"""
        return nn.Sequential(
            nn.Linear(3 + 1 + 1 + 8, 128),  # pos + freq + time + obj
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # E(3) + B(3) + scattering(1)
        )
    
    @staticmethod
    def physics_informed_mlp():
        """물리 법칙이 추가된 MLP"""
        class PhysicsMLPModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(13, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 7)
                )
            
            def forward(self, positions, frequencies, times, dynamic_objects):
                x = torch.cat([positions, frequencies, times, dynamic_objects], dim=1)
                output = self.mlp(x)
                
                return {
                    'electric_field': output[:, :3],
                    'magnetic_field': output[:, 3:6],
                    'scattering_coefficient': torch.sigmoid(output[:, 6:7]),
                    'propagation_delay': torch.abs(output[:, 6:7]) * 1e-9
                }
        
        return PhysicsMLPModel()

def run_comparative_benchmark():
    """다양한 모델들의 성능 비교"""
    logger.info("=== 모델 성능 비교 벤치마크 ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 테스트 데이터
    test_dataset = SyntheticEMDataset(n_samples=1000)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    models = {
        'EM-NeRF (Ours)': OptimizedEMNeRF(hidden_dim=256, n_layers=8),
        'Simple MLP': BaselineModels.simple_mlp(),
        'Physics MLP': BaselineModels.physics_informed_mlp()
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"벤치마킹: {name}")
        
        model = model.to(device)
        
        # 파라미터 수 계산
        n_params = sum(p.numel() for p in model.parameters())
        
        # 추론 시간 측정
        inference_times = []
        
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                positions = batch['position'].to(device)
                frequencies = batch['frequency'].to(device)
                times = batch['time'].to(device)
                dynamic_objects = batch['dynamic_objects'].to(device)
                
                torch.cuda.synchronize()
                start_time = time.time()
                
                if isinstance(model, nn.Sequential):
                    # Simple MLP 케이스
                    x = torch.cat([positions, frequencies, times, dynamic_objects], dim=1)
                    _ = model(x)
                else:
                    _ = model(positions, frequencies, times, dynamic_objects)
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
        
        results[name] = {
            'parameters': n_params,
            'mean_inference_time': np.mean(inference_times),
            'throughput': 32 / np.mean(inference_times)  # samples/sec
        }
    
    # 결과 출력
    print("\n" + "="*60)
    print("모델 성능 비교 결과")
    print("="*60)
    print(f"{'모델명':<20} {'파라미터':<15} {'추론시간(s)':<15} {'처리량(sps)':<15}")
    print("-"*60)
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['parameters']:<15,} {metrics['mean_inference_time']:<15.4f} {metrics['throughput']:<15.1f}")
    
    return results

# RunPod 실행 가이드 출력
def print_runpod_guide():
    """RunPod 사용 가이드 출력"""
    
    guide = """
🚀 RunPod GPU 인스턴스 실행 가이드
================================

1. RunPod 계정 생성 및 GPU 인스턴스 시작
   - https://runpod.io 접속
   - RTX 4090, A100, H100 등 선택 (권장: 12GB+ VRAM)
   - PyTorch 2.0+ 템플릿 선택

2. 파일 업로드
   - 이 Python 파일을 /workspace/ 디렉토리에 업로드
   - setup_runpod.sh 스크립트 실행으로 환경 설정

3. 실행 명령어
   ```bash
   cd /workspace
   chmod +x setup_runpod.sh
   ./setup_runpod.sh
   python runpod_em_nerf_test.py
   ```

4. 기대 성능 (RTX 4090 기준)
   - 훈련 속도: 50-100 epoch/분
   - 추론 속도: 1000+ samples/초
   - 메모리 사용량: 8-12GB

5. 결과 파일들
   - em_nerf_model_*.pth: 훈련된 모델
   - em_nerf_benchmark_*.json: 성능 리포트  
   - em_field_visualization_*.png: 시각화 결과

6. 비용 최적화 팁
   - Spot 인스턴스 사용으로 50% 비용 절약
   - 필요시에만 GPU 사용, 유휴시 정지
   - 데이터 전처리는 CPU 인스턴스에서 수행

💡 문제 해결
===========
- GPU 메모리 부족: 배치 크기 줄이기 (batch_size=16 또는 8)
- 느린 훈련: mixed precision 활성화 확인
- 데이터 로딩 느림: num_workers 조정

📊 실제 데이터셋 사용법
====================
1. NIST RF 데이터: https://its.ntia.gov/public-data/
2. DeepSig RadioML: https://www.deepsig.ai/datasets/
3. 자체 측정 데이터: convert_matlab_to_pytorch() 함수 사용

🔗 추가 리소스
=============
- NeRF 논문: https://arxiv.org/abs/2003.08934
- Instant-NGP: https://arxiv.org/abs/2201.05989
- RF 전파 모델링: https://www.itu.int/rec/R-REC-P/en
"""
    
    print(guide)

# 추가 실행 옵션들
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "guide":
            print_runpod_guide()
        elif command == "memory":
            memory_profiler()
        elif command == "compare":
            run_comparative_benchmark()
        elif command == "setup":
            create_runpod_setup_script()
            download_real_datasets()
        else:
            print("알 수 없는 명령어. 사용법: python script.py [guide|memory|compare|setup]")
    else:
        # 기본 실행: 전체 벤치마크
        main_results = run_full_benchmark()
        
        # 추가 분석 실행
        print("\n추가 분석 실행 중...")
        memory_results = memory_profiler()
        comparison_results = run_comparative_benchmark()
        
        print("\n🎉 모든 벤치마크 완료!")
        print("자세한 가이드를 보려면: python script.py guide")