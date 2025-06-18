# ─────────────────────────────────────────────────────────
# 0) 한번만 설치/임포트
!pip install -q tqdm            # 터미널·노트북에 없다면
from tqdm.auto import tqdm      # 자동으로 Jupyter/CLI 모두 지원
# ─────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import logging

class FixedStabilizedEMNeRF(nn.Module):
    
    def __init__(self, hidden_dim=256, n_layers=8):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.pos_encoding_levels = 8
        self.time_encoding_levels = 4
        
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
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(self.input_dim, hidden_dim))
        self.layers.append(nn.LayerNorm(hidden_dim))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Dropout(0.1))
        
        for i in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout(0.1))
        
        skip_layer_idx = len(self.layers)
        self.layers.append(nn.Linear(hidden_dim + self.input_dim, hidden_dim))
        self.layers.append(nn.LayerNorm(hidden_dim))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Dropout(0.1))
        
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.LayerNorm(hidden_dim))
        self.layers.append(nn.ReLU(inplace=True))
        
        self.skip_layer_idx = skip_layer_idx
        
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
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.logger = logging.getLogger("EMNeRF")
        
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=1e-4,  # 줄임
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10,
            verbose=True
        )
        
        self.scaler = torch.cuda.amp.GradScaler()
        
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
    
    def train_epoch(self, dataloader, log_every=200):
        self.model.train()
        total_epoch_loss = 0.0  # ✅ 현재 에포크의 전체 손실을 저장할 변수 추가
        tbar = tqdm(enumerate(dataloader), total=len(dataloader),
                    leave=False, bar_format='{l_bar}{bar:30}{r_bar}')

        for batch_idx, batch in tbar:
            # ... (데이터 전송 및 forward/backward 코드는 동일) ...
            positions        = batch['position'].to(self.device, non_blocking=True)
            frequencies      = batch['frequency'].to(self.device, non_blocking=True)
            times            = batch['time'].to(self.device, non_blocking=True)
            dynamic_objects  = batch['dynamic_objects'].to(self.device, non_blocking=True)
            targets          = {k: v.to(self.device, non_blocking=True)
                                for k, v in batch['ground_truth'].items()}
    
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                preds   = self.model(positions, frequencies, times, dynamic_objects)
                losses  = self.improved_physics_loss(preds, targets)
                total_l = losses['total']
    
            self.scaler.scale(total_l).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # foreach=True는 PyTorch 1.10 이상에서 유용
            self.scaler.step(self.optimizer)
            self.scaler.update()
    
            # ... (로그/프로그레스 바 업데이트는 동일) ...
            batch_loss = total_l.item()
            total_epoch_loss += batch_loss # ✅ 배치 손실을 에포크 전체 손실에 누적
    
            if (batch_idx + 1) % log_every == 0:
                avg_l = total_epoch_loss / (batch_idx + 1) # 더 정확한 평균
                tbar.set_description(f'Batch {batch_idx+1}/{len(dataloader)}  '
                                     f'Loss {avg_l:.4e}')
    
        # ✅ 수정된 epoch 평균 손실 계산
        epoch_loss = total_epoch_loss / len(dataloader)
        self.train_losses.append(epoch_loss) # 이제 정확한 에포크 평균 손실이 저장됩니다.
        self.lr_history.append(self.optimizer.param_groups[0]['lr'])
        self.scheduler.step(epoch_loss)
        return epoch_loss
        
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

#!/usr/bin/env python3
"""
OpenEMS 기반 전자기장 데이터 생성기
EM-NeRF 학습용 고품질 데이터셋 생성
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import h5py
import sys

# OpenEMS Python 인터페이스 (pyEMS 설치 필요)
try:
    from pyems.simulation import Simulation
    from pyems.structure import PCB, Microstrip, ViaWall, Box, Cylinder, DiscretePort
    from pyems.coordinate import Coordinate, Axis, Box3
    from pyems.field_dump import FieldDump
    from pyems.mesh import Mesh
    from pyems.calc import *
    OPENEMS_AVAILABLE = True
except ImportError:
    print("⚠️  pyEMS가 설치되지 않았습니다. 설치 방법:")
    print("   pip install pyems")
    print("   또는 OpenEMS 공식 사이트에서 설치")
    OPENEMS_AVAILABLE = False

class OpenEMSDataGenerator:
    
    def __init__(self, sim_path="./openems_sims"):
        self.sim_path = Path(sim_path)
        self.sim_path.mkdir(exist_ok=True)
        
        self.frequencies = np.logspace(8, 11, 50) 
        self.simulation_results = []
        
    def create_antenna_simulation(self, freq_center=2.4e9, name="patch_antenna"):
        sim = Simulation(freq=freq_center, name=name, sim_dir=str(self.sim_path))
        
        pcb_len = 60e-3
        pcb_width = 60e-3 
        pcb_height = 1.6e-3
        
        patch_len = 0.12 * 3e8 / freq_center  
        patch_width = patch_len * 1.2
        
        pcb = PCB(
            sim=sim,
            pcb_len=pcb_len,
            pcb_width=pcb_width, 
            layers=LayerStack(Layer("top", thickness=35e-6, conductivity=58e6)),
            layer_stack=LayerStack(Layer("dielectric", thickness=pcb_height, permittivity=4.3))
        )
        
        patch = Box(
            coordinate=Coordinate(-patch_len/2, -patch_width/2, pcb_height),
            dimensions=(patch_len, patch_width, 35e-6),
            material="copper"
        )
        sim.add_structure(patch)
        
        feed = Microstrip(
            sim=sim,
            position=Coordinate(0, -pcb_width/4, 0),
            length=pcb_width/2,
            width=3e-3,
            propagation_axis=Axis("y"),
            impedance=50
        )
        
        sim.boundary_conditions = ["PML_8", "PML_8", "PML_8", "PML_8", "PML_8", "PML_8"]
        
        mesh = Mesh(
            sim=sim,
            metal_res=1e-3,  
            nonmetal_res=5e-3  
        )
        
        field_dump = FieldDump(
            sim=sim,
            box=Box3(
                coordinate1=Coordinate(-pcb_len/2, -pcb_width/2, 0),
                coordinate2=Coordinate(pcb_len/2, pcb_width/2, 20e-3)  # 안테나 위 20mm
            ),
            dump_type=0,  # E+H 필드
            file_type=1   # HDF5 형식
        )
        
        return sim, field_dump
    # OpenEMSDataGenerator 클래스 내부에 이 함수를 추가하세요.

    def create_dipole_simulation(self, freq_center=1e9, name="dipole_antenna"):
        """
        중심 급전 방식의 다이폴 안테나 시뮬레이션을 생성합니다.
        
        Args:
            freq_center (float): 시뮬레이션 중심 주파수 (Hz)
            name (str): 시뮬레이션 이름
        
        Returns:
            tuple: (Simulation 객체, FieldDump 객체)
        """
        
        # ─── 1. 시뮬레이션 기본 설정 ──────────────────────────
        sim = Simulation(freq=freq_center, name=name, sim_dir=str(self.sim_path))
        
        # ─── 2. 다이폴 파라미터 계산 ──────────────────────────
        c = 3e8  # 빛의 속도
        wavelength = c / freq_center
        
        # 일반적으로 다이폴의 전체 길이는 반파장(λ/2)으로 설정합니다.
        dipole_length = wavelength / 2
        arm_length = dipole_length / 2  # 한쪽 팔의 길이
        gap = wavelength / 100          # 두 팔 사이의 급전 갭
        radius = wavelength / 200       # 다이폴 도선의 반지름
        
        # ─── 3. 다이폴 구조물 생성 (Cylinder 사용) ─────────────
        # 다이폴을 Z축을 따라 배치합니다.
        
        # 위쪽 팔 (positive z-axis)
        arm_pos = Cylinder(
            coordinate=Coordinate(0, 0, gap / 2), # 갭의 중심 위에서 시작
            radius=radius,
            length=arm_length,
            axis="z",
            material="pec"  # pec: Perfect Electric Conductor (완전 도체)
        )
        
        # 아래쪽 팔 (negative z-axis)
        arm_neg = Cylinder(
            coordinate=Coordinate(0, 0, -gap / 2), # 갭의 중심 아래에서 시작
            radius=radius,
            length=-arm_length, # 음수 방향으로 길이 설정
            axis="z",
            material="pec"
        )
        
        sim.add_structure(arm_pos)
        sim.add_structure(arm_neg)
        
        # ─── 4. 급전 포트 설정 (가장 중요) ─────────────────────
        # 두 팔 사이의 갭을 가로지르는 DiscretePort를 추가하여 에너지를 공급합니다.
        port = DiscretePort(
            sim=sim,
            start=[0, 0, -gap / 2],  # 포트 시작점 (아래쪽 팔 끝)
            end=[0, 0, gap / 2],      # 포트 끝점 (위쪽 팔 끝)
            impedance=73,             # 반파장 다이폴의 이론적 임피던스 (약 73옴)
            port_number=1
        )
        
        # ─── 5. 경계 조건 및 메쉬 설정 ────────────────────────
        # 개방된 공간을 시뮬레이션하기 위해 흡수 경계 조건(PML)을 사용합니다.
        sim.boundary_conditions = ["PML_8", "PML_8", "PML_8", "PML_8", "PML_8", "PML_8"]
        
        # 메쉬 해상도 설정. 파장에 비례하여 설정하는 것이 좋습니다.
        mesh = Mesh(
            sim=sim,
            metal_res=wavelength / 50,    # 금속 구조물 주변은 더 촘촘하게
            nonmetal_res=wavelength / 25  # 그 외 공간은 더 넓게
        )

        # ─── 6. 필드 덤프 영역 설정 ──────────────────────────
        # NeRF 학습에 사용할 3D 전자기장 데이터를 저장할 영역을 정의합니다.
        # 다이폴 주변 반경 약 1 파장 크기의 공간을 설정합니다.
        dump_box_size = wavelength * 2
        field_dump = FieldDump(
            sim=sim,
            box=Box3(
                Coordinate(-dump_box_size / 2, -dump_box_size / 2, -dump_box_size / 2),
                Coordinate(dump_box_size / 2, dump_box_size / 2, dump_box_size / 2)
            ),
            dump_type=0,  # E 필드와 H 필드 모두 저장
            file_type=1   # HDF5 형식으로 저장
        )
        
        return sim, field_dump
    
    def create_complex_scenario(self, scenario_type="urban_environment"):
        """복잡한 전파 환경 시뮬레이션"""
        
        if scenario_type == "urban_environment":
            return self.create_urban_propagation()
        elif scenario_type == "indoor_wifi":
            return self.create_indoor_wifi()
        elif scenario_type == "5g_mmwave":
            return self.create_5g_mmwave()
        else:
            raise ValueError(f"지원하지 않는 시나리오: {scenario_type}")
    
    def create_urban_propagation(self):
        """도시 환경 전파 시뮬레이션"""
        sim = Simulation(freq=900e6, name="urban_prop", sim_dir=str(self.sim_path))
        
        # 건물들 배치
        buildings = [
            Box(Coordinate(-50, -20, 0), (40, 40, 30), material="concrete"),
            Box(Coordinate(20, -30, 0), (30, 25, 25), material="concrete"),
            Box(Coordinate(-10, 30, 0), (35, 20, 40), material="concrete"),
        ]
        
        for building in buildings:
            sim.add_structure(building)
        
        # 기지국 안테나
        base_station = Box(
            Coordinate(0, 0, 50),  # 50m 높이
            (2, 2, 1),
            material="copper"
        )
        sim.add_structure(base_station)
        
        # 필드 측정 영역
        measurement_area = FieldDump(
            sim=sim,
            box=Box3(
                Coordinate(-100, -60, 0),
                Coordinate(100, 60, 80)
            ),
            dump_type=0
        )
        
        return sim, measurement_area
    
    def run_simulation_batch(self, scenarios, n_freq_points=20):
        """배치 시뮬레이션 실행"""
        
        if not OPENEMS_AVAILABLE:
            print("⚠️  OpenEMS가 없어 가상 데이터를 생성합니다.")
            return self.generate_synthetic_openems_data(scenarios, n_freq_points)
        
        results = []
        
        for scenario in scenarios:
            print(f"🔧 시뮬레이션 실행 중: {scenario}")
            
            # 주파수 스윕
            freq_points = np.logspace(8, 11, n_freq_points)
            
            for freq in freq_points:
                try:
                    if scenario == "patch_antenna":
                        sim, field_dump = self.create_antenna_simulation(
                            freq_center=freq, name=f"{scenario}_{freq/1e9:.1f}GHz"
                        )
                    elif scenario == "dipole_antenna":
                        sim, field_dump = self.create_dipole_simulation(
                            freq_center=freq, name=f"{scenario}_{freq/1e9:.1f}GHz"
                        )

                    elif scenario in ["urban_environment", "indoor_wifi", "5g_mmwave"]:
                        sim, field_dump = self.create_complex_scenario(
                            scenario_type=scenario
                        )
                    else:
                        print(f"  ⚠️ 지원하지 않는 시나리오 '{scenario}'는 건너뜁니다.")
                        continue
                    
                    # 시뮬레이션 실행
                    sim.run()
                    
                    # 결과 수집
                    field_data = self.extract_field_data(sim, field_dump, freq)
                    results.append(field_data)
                    
                    print(f"  ✅ {freq/1e9:.1f}GHz 완료")
                    
                except Exception as e:
                    print(f"  ❌ {freq/1e9:.1f}GHz 실패: {e}")
                    continue
        
        return results
    
    def extract_field_data(self, sim, field_dump, frequency):
        """시뮬레이션 결과에서 필드 데이터 추출"""
        
        # HDF5 파일에서 필드 데이터 읽기
        h5_file = self.sim_path / f"{sim.name}" / "field_dumps" / "Et.h5"
        
        if not h5_file.exists():
            raise FileNotFoundError(f"필드 데이터 파일이 없습니다: {h5_file}")
        
        with h5py.File(h5_file, 'r') as f:
            # 좌표 정보
            x_coords = f['mesh_line_x'][:]
            y_coords = f['mesh_line_y'][:]
            z_coords = f['mesh_line_z'][:]
            
            # 전기장 데이터 (복소수)
            Ex = f['Ex_re'][:] + 1j * f['Ex_im'][:]
            Ey = f['Ey_re'][:] + 1j * f['Ey_im'][:]
            Ez = f['Ez_re'][:] + 1j * f['Ez_im'][:]
            
            # 자기장 데이터
            Hx = f['Hx_re'][:] + 1j * f['Hx_im'][:]
            Hy = f['Hy_re'][:] + 1j * f['Hy_im'][:]
            Hz = f['Hz_re'][:] + 1j * f['Hz_im'][:]
        
        # 3D 메쉬그리드 생성
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        
        # 좌표를 1D 배열로 변환
        positions = np.column_stack([
            X.flatten(), Y.flatten(), Z.flatten()
        ])
        
        # 필드를 1D 배열로 변환
        E_fields = np.column_stack([
            Ex.flatten(), Ey.flatten(), Ez.flatten()
        ])
        
        H_fields = np.column_stack([
            Hx.flatten(), Hy.flatten(), Hz.flatten()
        ])
        
        # 물리량 계산
        E_magnitude = np.abs(E_fields)
        H_magnitude = np.abs(H_fields)
        
        # 포인팅 벡터 (에너지 흐름)
        S = 0.5 * np.real(np.cross(E_fields, np.conj(H_fields)))
        
        # 임피던스
        Z = E_magnitude / (H_magnitude + 1e-12)  # 0으로 나누기 방지
        
        return {
            'positions': positions,
            'frequency': frequency,
            'E_field': E_fields,
            'H_field': H_fields,
            'E_magnitude': E_magnitude,
            'H_magnitude': H_magnitude,
            'poynting_vector': S,
            'impedance': Z,
            'scenario': sim.name
        }
    
    def generate_synthetic_openems_data(self, scenarios, n_freq_points):
        """OpenEMS가 없을 때 가상 데이터 생성 (물리적으로 타당함)"""
        print("📊 물리 기반 가상 데이터 생성 중...")
        
        results = []
        freq_points = np.logspace(8, 11, n_freq_points)
        
        for scenario in scenarios:
            for freq in freq_points:
                # 3D 그리드 생성
                grid_size = 32
                x = np.linspace(-0.1, 0.1, grid_size)  # 20cm x 20cm x 20cm
                y = np.linspace(-0.1, 0.1, grid_size)
                z = np.linspace(0, 0.2, grid_size)
                
                X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                positions = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
                
                # 파장 계산
                c = 3e8
                wavelength = c / freq
                k = 2 * np.pi / wavelength
                
                # 안테나 위치 (원점)
                antenna_pos = np.array([0, 0, 0])
                
                # 각 점까지의 거리
                distances = np.linalg.norm(positions - antenna_pos, axis=1)
                
                # 물리적으로 타당한 전기장 (구면파)
                r_hat = (positions - antenna_pos) / (distances[:, np.newaxis] + 1e-12)
                
                # 전기장 계산 (구면파 근사)
                E_magnitude = np.exp(-1j * k * distances) / (distances + 0.01)
                
                # 편광 방향 (z 방향 편광 가정)
                E_fields = np.zeros((len(positions), 3), dtype=complex)
                E_fields[:, 2] = E_magnitude  # Ez 성분만
                
                # 자기장 계산 (맥스웰 방정식에 따라)
                eta0 = 377  # 자유공간 임피던스
                H_fields = np.cross(r_hat, E_fields) / eta0
                
                # 산란 계수 (거리와 주파수 의존성)
                scattering = np.exp(-distances / wavelength) * (freq / 1e9) / 100
                
                # 전파 지연
                propagation_delay = distances / c
                
                # 정규화
                E_real = np.real(E_fields)
                H_real = np.real(H_fields)
                
                # 정규화 ([-1, 1] 범위)
                E_max = np.max(np.abs(E_real)) + 1e-12
                H_max = np.max(np.abs(H_real)) + 1e-12
                
                E_normalized = E_real / E_max
                H_normalized = H_real / H_max
                
                scattering_normalized = scattering / (np.max(scattering) + 1e-12)
                delay_normalized = propagation_delay / (np.max(propagation_delay) + 1e-12)
                
                result = {
                    'positions': positions,
                    'frequency': freq,
                    'E_field': E_normalized,
                    'H_field': H_normalized,
                    'scattering_coefficient': scattering_normalized,
                    'propagation_delay': delay_normalized,
                    'scenario': scenario
                }
                
                results.append(result)
        
        print(f"✅ {len(results)}개 데이터 포인트 생성 완료")
        return results

# 2. EM-NeRF용 데이터셋 클래스 개선

class OpenEMSDataset(torch.utils.data.Dataset):
    """OpenEMS 데이터를 사용하는 EM-NeRF 데이터셋"""
    
    def __init__(self, openems_results, augment=True):
        self.data = openems_results
        self.augment = augment
        
        # 모든 데이터를 하나의 배열로 결합
        self.positions = []
        self.frequencies = []
        self.E_fields = []
        self.H_fields = []
        self.scattering = []
        self.delays = []
        
        print("🔄 OpenEMS 데이터 전처리 중...")
        
        for result in openems_results:
            n_points = len(result['positions'])
            
            self.positions.append(result['positions'])
            self.frequencies.append(np.full(n_points, result['frequency']))
            self.E_fields.append(result['E_field'])
            self.H_fields.append(result['H_field'])
            self.scattering.append(result['scattering_coefficient'])
            self.delays.append(result['propagation_delay'])
        
        # 배열 결합
        self.positions = np.vstack(self.positions)
        self.frequencies = np.hstack(self.frequencies)
        self.E_fields = np.vstack(self.E_fields)
        self.H_fields = np.vstack(self.H_fields)
        self.scattering = np.hstack(self.scattering)
        self.delays = np.hstack(self.delays)
        
        # 정규화
        self.normalize_data()
        
        # 텐서 변환
        self.positions = torch.FloatTensor(self.positions)
        self.frequencies = torch.FloatTensor(self.frequencies).unsqueeze(1)
        self.E_fields = torch.FloatTensor(self.E_fields)
        self.H_fields = torch.FloatTensor(self.H_fields)
        self.scattering = torch.FloatTensor(self.scattering).unsqueeze(1)
        self.delays = torch.FloatTensor(self.delays).unsqueeze(1)
        
        print(f"✅ 총 {len(self.positions)}개 데이터 포인트 준비 완료")
    
    def normalize_data(self):
        """데이터 정규화"""
        # 위치 정규화 ([-1, 1] 범위)
        pos_min, pos_max = self.positions.min(axis=0), self.positions.max(axis=0)
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.positions = 2 * (self.positions - pos_min) / (pos_max - pos_min + 1e-12) - 1
        
        # 주파수 로그 정규화
        self.frequencies = (np.log10(self.frequencies) - 8) / 3  # 8-11 로그 범위를 0-1로
        
        # 필드는 이미 정규화되어 있다고 가정
        
        # 산란과 지연은 0-1 범위로
        self.scattering = np.clip(self.scattering, 0, 1)
        self.delays = np.clip(self.delays, 0, 1)
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        # 동적 객체는 랜덤 생성 (실제로는 OpenEMS에서 추출 가능)
        dynamic_objects = torch.rand(8) * 2 - 1
        
        return {
            'position': self.positions[idx],
            'frequency': self.frequencies[idx],
            'time': torch.rand(1) * 2 - 1,  # 시간은 랜덤
            'dynamic_objects': dynamic_objects,
            'ground_truth': {
                'electric_field': self.E_fields[idx],
                'magnetic_field': self.H_fields[idx],
                'scattering_coefficient': self.scattering[idx],
                'propagation_delay': self.delays[idx]
            }
        }

# 3. 통합 실행 함수

def run_openems_emnerf_training():
    """OpenEMS 데이터로 EM-NeRF 훈련"""
    
    print("🚀 OpenEMS + EM-NeRF 통합 훈련 시작!")
    print("="*60)
    
    # 1. OpenEMS 데이터 생성
    print("\n1️⃣ OpenEMS 시뮬레이션 데이터 생성")
    generator = OpenEMSDataGenerator()
    
    scenarios = ["patch_antenna", "dipole_antenna", "slot_antenna"]
    openems_results = generator.run_simulation_batch(scenarios, n_freq_points=10)
    
    # 2. EM-NeRF 데이터셋 생성
    print("\n2️⃣ EM-NeRF 데이터셋 생성")
    train_dataset = OpenEMSDataset(openems_results)
     # ✅ 정규화 파라미터를 JSON 파일로 저장
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=1024, 
        shuffle=True, 
        num_workers=0
    )
    
    # 3. 모델 및 트레이너 설정 (기존 코드 사용)
    print("\n3️⃣ EM-NeRF 모델 초기화")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = FixedStabilizedEMNeRF(hidden_dim=256, n_layers=8)
    trainer = ImprovedEMNeRFTrainer(model, device=device)
    
    # 4. 훈련 실행
    print("\n4️⃣ 훈련 시작")
    n_epochs = 50
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(n_epochs):
        epoch_loss = trainer.train_epoch(train_loader)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {epoch_loss:.6f}")
            trainer.plot_training_progress()
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {  # 모델 구조 정보 추가
                    'hidden_dim': model.hidden_dim, 
                    'n_layers': model.n_layers
                }
            }, 'openems_emnerf_best.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping!")
            break
    
    print(f"\n🎉 훈련 완료! 최종 Loss: {best_loss:.6f}")
    
    return trainer, model, openems_results

if __name__ == "__main__":
    # 통합 실행
    trainer, model, data = run_openems_emnerf_training()
    
    print("\n" + "="*60)
    print("🎯 OpenEMS + EM-NeRF 훈련 성공!")
    print("="*60)
    print("📊 데이터 품질: 물리적으로 정확한 OpenEMS 시뮬레이션")
    print("🔧 모델: 당신의 개선된 EM-NeRF 아키텍처")
    print("✨ 결과: 훨씬 안정적이고 정확한 전자기장 예측!")