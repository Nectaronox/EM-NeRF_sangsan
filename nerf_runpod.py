#!/usr/bin/env python3
"""
개선된 전자기장 NeRF - Loss 수렴 문제 해결
주요 개선사항:
1. 적절한 스케일링 및 정규화
2. Gradient clipping 추가
3. 개선된 물리 제약 조건
4. 더 안정적인 훈련 파라미터
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedSyntheticEMDataset(Dataset):
    """개선된 합성 전자기장 데이터셋 - 적절한 스케일링 적용"""
    
    def __init__(self, n_samples=50000, grid_size=64, frequency_range=(1e9, 10e9)):
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.frequency_range = frequency_range
        
        logger.info(f"개선된 데이터셋 생성 중: {n_samples} 샘플")
        
        # 3D 좌표 생성 (정규화됨)
        self.positions = torch.rand(n_samples, 3) * 2 - 1  # [-1, 1] 범위
        
        # 주파수 생성 및 정규화
        raw_frequencies = torch.rand(n_samples, 1) * (frequency_range[1] - frequency_range[0]) + frequency_range[0]
        self.frequencies = (torch.log10(raw_frequencies) - 9) / 1  # [0, 1] 범위로 정규화
        
        # 시간 생성 및 정규화
        self.times = torch.rand(n_samples, 1) * 2 - 1  # [-1, 1] 범위
        
        # 동적 객체 정보 (정규화됨)
        self.dynamic_objects = self.generate_dynamic_objects(n_samples)
        
        # 개선된 Ground truth 전자기장 (적절한 스케일링)
        self.em_fields = self.simulate_improved_ground_truth()
        
        logger.info("개선된 데이터셋 생성 완료")
    
    def generate_dynamic_objects(self, n_samples):
        """정규화된 동적 객체 정보 생성"""
        objects = torch.zeros(n_samples, 8)
        
        # 위치 ([-1, 1] 범위)
        objects[:, :3] = torch.rand(n_samples, 3) * 2 - 1
        
        # 속도 정규화 ([-1, 1] 범위)
        objects[:, 3:6] = torch.rand(n_samples, 3) * 2 - 1
        
        # 객체 타입 정규화 (0~1 범위)
        objects[:, 6] = torch.rand(n_samples)
        
        # 크기 정규화 (0~1 범위)
        objects[:, 7] = torch.rand(n_samples)
        
        return objects
    
    def simulate_improved_ground_truth(self):
        """개선된 물리 기반 ground truth - 적절한 스케일링"""
        logger.info("개선된 Ground truth 계산 중...")
        
        n_samples = len(self.positions)
        
        # 모든 값들을 [-1, 1] 또는 [0, 1] 범위로 정규화
        E_field = torch.zeros(n_samples, 3)
        B_field = torch.zeros(n_samples, 3)
        scattering = torch.zeros(n_samples, 1)
        delay = torch.zeros(n_samples, 1)
        
        for i in range(n_samples):
            pos = self.positions[i]
            freq_norm = self.frequencies[i].item()  # 이미 정규화됨
            obj = self.dynamic_objects[i]
            
            # 거리 기반 감쇠 (정규화됨)
            distance = torch.norm(pos).clamp(min=0.1)
            
            # 정규화된 전기장 ([-1, 1] 범위)
            E_field[i, 0] = torch.tanh(torch.sin(2 * np.pi * distance + freq_norm * np.pi))
            E_field[i, 1] = torch.tanh(torch.cos(2 * np.pi * distance + freq_norm * np.pi))
            E_field[i, 2] = torch.tanh(torch.sin(4 * np.pi * distance + freq_norm * np.pi))
            
            # 자기장 (E와 일관성 있게, 스케일 조정됨)
            B_field[i, 0] = E_field[i, 1] * 0.1  # 물리적 관계 유지하되 스케일 조정
            B_field[i, 1] = -E_field[i, 0] * 0.1
            B_field[i, 2] = E_field[i, 2] * 0.1
            
            # 산란 계수 (0~1 범위)
            obj_influence = torch.norm(obj[:3] - pos)
            scattering[i] = torch.sigmoid(2 * (obj[7] - obj_influence))
            
            # 전파 지연 (0~1 범위)
            delay[i] = torch.sigmoid(distance + scattering[i])
        
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

class FixedStabilizedEMNeRF(nn.Module):
    """차원 문제를 해결한 안정화된 전자기장 NeRF"""
    
    def __init__(self, hidden_dim=256, n_layers=8):
        super().__init__()
        
        # 위치 인코딩 레벨
        self.pos_encoding_levels = 8
        self.time_encoding_levels = 4
        
        # 입력 차원 정확히 계산
        pos_dim = 3 + 3 * 2 * self.pos_encoding_levels  # 3 + 48 = 51
        time_dim = 1 + 1 * 2 * self.time_encoding_levels  # 1 + 8 = 9
        freq_dim = 1 + 1 * 2 * 3  # 1 + 6 = 7 (frequency_encoding에서)
        obj_dim = 8
        
        self.input_dim = pos_dim + time_dim + freq_dim + obj_dim  # 51 + 9 + 7 + 8 = 75
        
        print(f"계산된 입력 차원: {self.input_dim}")
        print(f"  - 위치 인코딩: {pos_dim}")
        print(f"  - 시간 인코딩: {time_dim}")
        print(f"  - 주파수 인코딩: {freq_dim}")
        print(f"  - 동적 객체: {obj_dim}")
        
        # 단순하고 안전한 네트워크 구조
        self.layers = nn.ModuleList()
        
        # 첫 번째 레이어
        self.layers.append(nn.Linear(self.input_dim, hidden_dim))
        self.layers.append(nn.LayerNorm(hidden_dim))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Dropout(0.1))
        
        # 중간 레이어들 (skip connection 없이 단순화)
        for i in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout(0.1))
        
        # Skip connection용 레이어 (중간에 추가)
        skip_layer_idx = len(self.layers)
        self.layers.append(nn.Linear(hidden_dim + self.input_dim, hidden_dim))
        self.layers.append(nn.LayerNorm(hidden_dim))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Dropout(0.1))
        
        # 마지막 레이어
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.LayerNorm(hidden_dim))
        self.layers.append(nn.ReLU(inplace=True))
        
        # Skip connection이 적용될 레이어 인덱스 저장
        self.skip_layer_idx = skip_layer_idx
        
        # 출력 헤드들
        self.electric_field_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Tanh()
        )
        
        self.magnetic_field_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Tanh()
        )
        
        self.scattering_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.delay_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Xavier 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def positional_encoding(self, x, levels):
        encoded = [x]
        for level in range(levels):
            for fn in [torch.sin, torch.cos]:
                encoded.append(fn(2**level * np.pi * x))
        return torch.cat(encoded, dim=-1)
    
    def frequency_encoding(self, freq):
        # 이미 정규화된 주파수 사용
        encoded = [freq]
        for i in range(3):
            encoded.append(torch.sin(2**i * np.pi * freq))
            encoded.append(torch.cos(2**i * np.pi * freq))
        return torch.cat(encoded, dim=-1)
    
    def forward(self, positions, frequencies, times, dynamic_objects):
        # 인코딩
        pos_encoded = self.positional_encoding(positions, self.pos_encoding_levels)
        time_encoded = self.positional_encoding(times, self.time_encoding_levels)
        freq_encoded = self.frequency_encoding(frequencies)
        obj_normalized = dynamic_objects
        
        # 특징 결합
        features = torch.cat([pos_encoded, time_encoded, freq_encoded, obj_normalized], dim=1)
        
        # 차원 확인
        assert features.shape[1] == self.input_dim, f"입력 차원 불일치: {features.shape[1]} != {self.input_dim}"
        
        # 네트워크 통과
        x = features
        skip_input = features
        
        for i, layer in enumerate(self.layers):
            # Skip connection 적용
            if i == self.skip_layer_idx:
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

class ImprovedEMNeRFTrainer:
    """개선된 NeRF 트레이너 - 안정적인 훈련"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # 더 보수적인 학습률
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=1e-4,  # 줄임
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # 스케줄러
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10,
            verbose=True
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        
        # 훈련 기록
        self.train_losses = []
        self.lr_history = []
    
    def improved_physics_loss(self, predictions, targets):
        """개선된 물리 기반 손실 함수"""
        mse_loss = nn.MSELoss()
        
        # 기본 MSE 손실들
        e_loss = mse_loss(predictions['electric_field'], targets['electric_field'])
        b_loss = mse_loss(predictions['magnetic_field'], targets['magnetic_field'])
        s_loss = mse_loss(predictions['scattering_coefficient'], targets['scattering_coefficient'])
        d_loss = mse_loss(predictions['propagation_delay'], targets['propagation_delay'])
        
        # 개선된 물리 제약 조건
        E_magnitude = torch.norm(predictions['electric_field'], dim=1)
        B_magnitude = torch.norm(predictions['magnetic_field'], dim=1)
        
        # 정규화된 물리 제약 (|B| ≈ |E|/c, 하지만 정규화된 스케일)
        # E와 B 모두 [-1,1] 범위이므로 단순한 비례 관계 사용
        physics_constraint = mse_loss(B_magnitude, E_magnitude * 0.1)  # 스케일 조정
        
        # E와 B의 직교성 제약 (간단화)
        dot_product = torch.sum(predictions['electric_field'] * predictions['magnetic_field'], dim=1)
        orthogonality_constraint = torch.mean(dot_product**2)
        
        # 가중 합산
        total_loss = (
            e_loss + 
            b_loss + 
            s_loss + 
            d_loss + 
            0.01 * physics_constraint +  # 가중치 줄임
            0.01 * orthogonality_constraint
        )
        
        return {
            'total': total_loss,
            'electric': e_loss,
            'magnetic': b_loss,
            'scattering': s_loss,
            'delay': d_loss,
            'physics': physics_constraint,
            'orthogonality': orthogonality_constraint
        }
    
    def train_epoch(self, dataloader):
        self.model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            # 데이터 GPU로 이동
            positions = batch['position'].to(self.device)
            frequencies = batch['frequency'].to(self.device)
            times = batch['time'].to(self.device)
            dynamic_objects = batch['dynamic_objects'].to(self.device)
            
            targets = {k: v.to(self.device) for k, v in batch['ground_truth'].items()}
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                predictions = self.model(positions, frequencies, times, dynamic_objects)
                losses = self.improved_physics_loss(predictions, targets)
                total_loss = losses['total']
            
            # Backward pass with gradient clipping
            self.scaler.scale(total_loss).backward()
            
            # Gradient clipping 적용
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            epoch_losses.append(total_loss.item())
            
            # 정기적인 로깅
            if batch_idx % 50 == 0:
                logger.info(f"Batch {batch_idx}, Loss: {total_loss.item():.6f}")
                
                # 상세 손실 출력
                if batch_idx % 200 == 0:
                    logger.info(f"  E-field: {losses['electric'].item():.6f}")
                    logger.info(f"  B-field: {losses['magnetic'].item():.6f}")
                    logger.info(f"  Scattering: {losses['scattering'].item():.6f}")
                    logger.info(f"  Delay: {losses['delay'].item():.6f}")
                    logger.info(f"  Physics: {losses['physics'].item():.6f}")
        
        avg_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_loss)
        self.lr_history.append(self.optimizer.param_groups[0]['lr'])
        
        # 스케줄러 업데이트
        self.scheduler.step(avg_loss)
        
        return avg_loss
    
    def plot_training_progress(self):
        """훈련 진행 상황 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss 플롯
        ax1.semilogy(self.train_losses)
        ax1.set_title('Training Loss (Log Scale)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Learning rate 플롯
        ax2.semilogy(self.lr_history)
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()

def run_improved_training():
    """개선된 훈련 실행"""
    logger.info("=== 개선된 전자기장 NeRF 훈련 시작 ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"사용 디바이스: {device}")
    
    # 개선된 데이터셋
    train_dataset = ImprovedSyntheticEMDataset(n_samples=5000)  # 작게 시작
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    
    # 모델 및 트레이너
    model = FixedStabilizedEMNeRF(hidden_dim=128, n_layers=6)  # 수정된 클래스 사용
    trainer = ImprovedEMNeRFTrainer(model, device=device)
    
    # 훈련 실행
    n_epochs = 100
    best_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    logger.info("훈련 시작...")
    for epoch in range(n_epochs):
        start_time = time.time()
        epoch_loss = trainer.train_epoch(train_loader)
        epoch_time = time.time() - start_time
        
        logger.info(f"Epoch {epoch+1}/{n_epochs}")
        logger.info(f"  Loss: {epoch_loss:.8f}")
        logger.info(f"  Time: {epoch_time:.2f}s")
        logger.info(f"  LR: {trainer.optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            # 최고 모델 저장
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
        
        # 진행 상황 시각화 (매 20 에포크)
        if (epoch + 1) % 20 == 0:
            trainer.plot_training_progress()
    
    logger.info(f"훈련 완료! 최종 Loss: {best_loss:.8f}")
    
    return trainer, model

if __name__ == "__main__":
    # 개선된 훈련 실행
    trainer, model = run_improved_training()
    
    # 최종 결과 출력
    print("\n" + "="*50)
    print("🎯 개선된 훈련 완료!")
    print("="*50)
    print(f"✅ 최종 Loss: {min(trainer.train_losses):.8f}")
    print(f"📉 Loss 개선: {trainer.train_losses[0]:.3e} → {min(trainer.train_losses):.3e}")
    print(f"📊 총 에포크: {len(trainer.train_losses)}")
    
    # 기대 Loss 범위 안내
    print("\n📋 정상적인 Loss 범위:")
    print("  • 초기: 1e-1 ~ 1e0")
    print("  • 중간: 1e-2 ~ 1e-1") 
    print("  • 수렴: 1e-3 ~ 1e-2")
    print("  • 과적합 의심: < 1e-4")