import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import math

class ImprovedHashEncoding(nn.Module):
    
    def __init__(self, 
                 n_levels: int = 16,
                 n_features_per_level: int = 2,
                 log2_hashmap_size: int = 19,
                 base_resolution: int = 16,
                 finest_resolution: int = 512,
                 input_dim: int = 3):
        super().__init__()
        
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.input_dim = input_dim
        self.b = math.exp((math.log(finest_resolution) - math.log(base_resolution)) / max(1, n_levels - 1))
        
        self.hash_tables = nn.ModuleList([
            nn.Embedding(2**log2_hashmap_size, n_features_per_level)
            for _ in range(n_levels)
        ])
        
        for table in self.hash_tables:
            nn.init.xavier_uniform_(table.weight, gain=1e-4)
    
    def hash_function(self, coords: torch.Tensor) -> torch.Tensor:
        primes = torch.tensor([73856093, 19349663, 83492791], 
                            device=coords.device, dtype=torch.long)
        
        coords_int = torch.clamp(coords, 0, 2**20).long()
        
        hash_val = torch.zeros(coords.shape[0], device=coords.device, dtype=torch.long)
        for i in range(min(3, coords.shape[1])):
            hash_val = (hash_val + coords_int[:, i] * primes[i]) % (2**self.log2_hashmap_size)
        
        return hash_val
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0.0, 1.0)
        
        encoded_features = []
        
        for level in range(self.n_levels):
            resolution = min(int(self.base_resolution * (self.b ** level)), 
                           self.finest_resolution)
            
            scaled_coords = x * (resolution - 1)
            coords_floor = torch.floor(scaled_coords).long()
            coords_ceil = torch.clamp(coords_floor + 1, 0, resolution - 1)
            
            coords_floor = torch.clamp(coords_floor, 0, resolution - 1)
            weights = scaled_coords - coords_floor.float()
            
            corner_features = []
            for dx in [0, 1]:
                for dy in [0, 1]:
                    for dz in [0, 1]:
                        corner_coords = coords_floor.clone()
                        corner_coords[:, 0] = coords_floor[:, 0] if dx == 0 else coords_ceil[:, 0]
                        corner_coords[:, 1] = coords_floor[:, 1] if dy == 0 else coords_ceil[:, 1]
                        corner_coords[:, 2] = coords_floor[:, 2] if dz == 0 else coords_ceil[:, 2]
                        
                        hash_indices = self.hash_function(corner_coords)
                        features = self.hash_tables[level](hash_indices)
                        corner_features.append(features)
            
            w = weights
            interpolated = (corner_features[0] * (1-w[:,0:1]) * (1-w[:,1:2]) * (1-w[:,2:3]) +
                          corner_features[1] * w[:,0:1] * (1-w[:,1:2]) * (1-w[:,2:3]) +
                          corner_features[2] * (1-w[:,0:1]) * w[:,1:2] * (1-w[:,2:3]) +
                          corner_features[3] * w[:,0:1] * w[:,1:2] * (1-w[:,2:3]) +
                          corner_features[4] * (1-w[:,0:1]) * (1-w[:,1:2]) * w[:,2:3] +
                          corner_features[5] * w[:,0:1] * (1-w[:,1:2]) * w[:,2:3] +
                          corner_features[6] * (1-w[:,0:1]) * w[:,1:2] * w[:,2:3] +
                          corner_features[7] * w[:,0:1] * w[:,1:2] * w[:,2:3])
            
            encoded_features.append(interpolated)
        
        return torch.cat(encoded_features, dim=1)

class FrequencyDependentElectromagneticNeRF(nn.Module): 
    
    def __init__(self,
                 hidden_dim: int = 128,
                 n_layers: int = 4,
                 encoding_config: Optional[dict] = None,
                 frequency_bands: int = 10):
        super().__init__()
        
        if encoding_config is None:
            encoding_config = {}
        self.position_encoder = ImprovedHashEncoding(**encoding_config)
        
        self.frequency_bands = frequency_bands
        self.frequency_encoder = nn.Sequential(
            nn.Linear(1, frequency_bands * 2),  # sin, cos encoding
            nn.ReLU()
        )
        
        pos_encoding_dim = (self.position_encoder.n_levels * 
                           self.position_encoder.n_features_per_level)
        freq_encoding_dim = frequency_bands * 2
        
        self.backbone_layers = nn.ModuleList()
        input_dim = pos_encoding_dim + freq_encoding_dim
        
        for i in range(n_layers):
            if i == n_layers // 2:
                self.backbone_layers.append(nn.Linear(input_dim + hidden_dim, hidden_dim))
            else:
                self.backbone_layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            input_dim = hidden_dim
        
        self.electric_field_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Tanh()
        )
        
        self.magnetic_field_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Tanh()
        )
        
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 6)
        )
        
        self.material_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        self.source_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)
        )
    
    def encode_frequency(self, frequency: torch.Tensor) -> torch.Tensor:
        log_freq = torch.log10(torch.clamp(frequency, min=1e6, max=1e12))
        normalized_freq = (log_freq - 6) / 6
        
        freq_encoded = []
        for i in range(self.frequency_bands):
            freq_encoded.append(torch.sin(2**i * math.pi * normalized_freq))
            freq_encoded.append(torch.cos(2**i * math.pi * normalized_freq))
        
        return torch.cat(freq_encoded, dim=-1)
    
    def forward(self, x: torch.Tensor, frequency: torch.Tensor, time: float = 0.0) -> Dict:
        
        pos_encoded = self.position_encoder(x)
        
        freq_encoded = self.encode_frequency(frequency)
        
        features = torch.cat([pos_encoded, freq_encoded], dim=1)
        
        skip_connection = None
        for i, layer in enumerate(self.backbone_layers):
            if i == len(self.backbone_layers) // 2 and skip_connection is not None:
                features = torch.cat([features, skip_connection], dim=1)
            elif i == 0:
                skip_connection = features
            
            features = F.relu(layer(features))
        
        E_field_amplitude = self.electric_field_head(features)
        B_field_amplitude = self.magnetic_field_head(features)
        phases = self.phase_head(features)
        material_props = self.material_head(features)
        sources = self.source_head(features)
        
        omega = 2 * math.pi * frequency
        time_factor = torch.cos(omega * time + phases[:, :3])
        time_factor_B = torch.cos(omega * time + phases[:, 3:])
        
        E_field = E_field_amplitude * time_factor
        B_field = B_field_amplitude * time_factor_B
        
        epsilon_r = 1.0 + torch.sigmoid(material_props[:, 0:1]) * 10
        mu_r = 1.0 + torch.sigmoid(material_props[:, 1:2]) * 2
        conductivity = torch.sigmoid(material_props[:, 2:3]) * 1e-2
        
        return {
            'electric_field': E_field,
            'magnetic_field': B_field,
            'epsilon_r': epsilon_r,
            'mu_r': mu_r,
            'conductivity': conductivity,
            'charge_density': sources[:, 0:1],
            'current_density': sources[:, 1:4],
            'frequency': frequency,
            'omega': omega
        }

class ImprovedMaxwellPhysicsLoss(nn.Module):
    
    def __init__(self, 
                 c: float = 2.998e8,
                 epsilon_0: float = 8.854e-12,
                 mu_0: float = 4*math.pi*1e-7):
        super().__init__()
        self.c = c
        self.epsilon_0 = epsilon_0
        self.mu_0 = mu_0
        
    def compute_derivatives(self, 
                          field: torch.Tensor, 
                          positions: torch.Tensor,
                          create_graph: bool = True) -> torch.Tensor:
        if not positions.requires_grad:
            positions.requires_grad_(True)
            
        gradients = []
        for i in range(field.shape[1]):
            try:
                grad = torch.autograd.grad(
                    outputs=field[:, i],
                    inputs=positions,
                    grad_outputs=torch.ones_like(field[:, i]),
                    create_graph=create_graph,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                
                if grad is None:
                    grad = torch.zeros_like(positions)
                gradients.append(grad)
            except RuntimeError:
                grad = self.finite_difference_gradient(field[:, i], positions)
                gradients.append(grad)
                
        return torch.stack(gradients, dim=1)
    
    def finite_difference_gradient(self, 
                                 field: torch.Tensor, 
                                 positions: torch.Tensor,
                                 eps: float = 1e-6) -> torch.Tensor:
        grad = torch.zeros_like(positions)
        
        for dim in range(positions.shape[1]):
            pos_plus = positions.clone()
            pos_minus = positions.clone()
            pos_plus[:, dim] += eps
            pos_minus[:, dim] -= eps
            
            grad[:, dim] = torch.zeros_like(field)
            
        return grad
    
    def compute_curl(self, field: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        grad = self.compute_derivatives(field, positions)
        
        curl_x = grad[:, 2, 1] - grad[:, 1, 2]
        curl_y = grad[:, 0, 2] - grad[:, 2, 0]
        curl_z = grad[:, 1, 0] - grad[:, 0, 1]
        
        return torch.stack([curl_x, curl_y, curl_z], dim=1)
    
    def compute_divergence(self, field: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        grad = self.compute_derivatives(field, positions)
        return grad[:, 0, 0] + grad[:, 1, 1] + grad[:, 2, 2]
    
    def forward(self, predictions: Dict, positions: torch.Tensor) -> Dict:  
        E = predictions['electric_field']
        B = predictions['magnetic_field']
        rho = predictions['charge_density'].squeeze(-1)
        J = predictions['current_density']
        epsilon_r = predictions['epsilon_r'].squeeze(-1)
        mu_r = predictions['mu_r'].squeeze(-1)
        omega = predictions['omega']
        
        epsilon = epsilon_r * self.epsilon_0
        mu = mu_r * self.mu_0
        
        div_E = self.compute_divergence(E, positions)
        div_B = self.compute_divergence(B, positions)
        curl_E = self.compute_curl(E, positions)
        curl_B = self.compute_curl(B, positions)
        
        gauss_electric = div_E * epsilon - rho / self.epsilon_0
        
        gauss_magnetic = div_B
        
        faraday = curl_E + 1j * omega.unsqueeze(-1) * B
        
        H = B / mu.unsqueeze(-1)
        curl_H = self.compute_curl(H, positions)
        ampere = curl_H - J - 1j * omega.unsqueeze(-1) * epsilon.unsqueeze(-1) * E
        
        loss_gauss_E = torch.mean(gauss_electric**2)
        loss_gauss_B = torch.mean(gauss_magnetic**2)
        loss_faraday = torch.mean(torch.abs(faraday)**2)
        loss_ampere = torch.mean(torch.abs(ampere)**2)
        
        boundary_mask = (torch.abs(positions).max(dim=1)[0] > 0.9)
        if boundary_mask.any():
            E_boundary = E[boundary_mask]
            loss_boundary = torch.mean(E_boundary**2)
        else:
            loss_boundary = torch.tensor(0.0, device=E.device)
        
        return {
            'gauss_electric': loss_gauss_E,
            'gauss_magnetic': loss_gauss_B,
            'faraday': loss_faraday,
            'ampere': loss_ampere,
            'boundary': loss_boundary,
            'total_physics': (loss_gauss_E + loss_gauss_B + 
                            loss_faraday + loss_ampere + loss_boundary)
        }

class Image2EMFieldEstimator(nn.Module):
    
    def __init__(self, backbone_name='resnet50'):
        super().__init__()
        import torchvision.models as models
        
        if backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            self.backbone.fc = nn.Identity()
            feature_dim = 2048

        
        self.depth_head = nn.Sequential(
            nn.ConvTranspose2d(feature_dim//16, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Sigmoid()
        )
        
        self.material_head = nn.Sequential(
            nn.ConvTranspose2d(feature_dim//16, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 5, 4, 2, 1),
            nn.Softmax(dim=1)
        )
        
        self.spatial_mapper = nn.Sequential(
            nn.Linear(feature_dim + 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, image: torch.Tensor, camera_params: dict) -> dict:
        
        features = self.backbone(image)
        
        feature_map = features.view(features.size(0), -1, 7, 7)
        depth_map = self.depth_head(feature_map)
        material_map = self.material_head(feature_map)
        
        return {
            'features': features,
            'depth_map': depth_map,
            'material_map': material_map
        }

class RealTimeEMOptimizer:
    
    def __init__(self, em_model, image_estimator=None):
        self.em_model = em_model
        self.image_estimator = image_estimator
        self.measurement_buffer = []
        self.optimization_history = []
    
    def add_rf_measurement(self, position: torch.Tensor, 
                          power: float, frequency: float, timestamp: float):
        measurement = {
            'position': position,
            'power': power,
            'frequency': frequency,
            'timestamp': timestamp
        }
        self.measurement_buffer.append(measurement)
        
        if len(self.measurement_buffer) > 1000:
            self.measurement_buffer.pop(0)
    
    def update_model_with_measurements(self):
        if len(self.measurement_buffer) < 10:
            return
            
        recent_measurements = self.measurement_buffer[-50:]
        
        positions = torch.stack([m['position'] for m in recent_measurements])
        powers = torch.tensor([m['power'] for m in recent_measurements])
        frequencies = torch.tensor([[m['frequency']] for m in recent_measurements])
        
        predictions = self.em_model(positions, frequencies)
        predicted_power = torch.sum(predictions['electric_field']**2, dim=1)
        
        measurement_loss = F.mse_loss(predicted_power, powers)
        
        optimizer = torch.optim.Adam(self.em_model.parameters(), lr=1e-5)
        optimizer.zero_grad()
        measurement_loss.backward()
        optimizer.step()

class CommunicationSystemOptimizer:
    
    def __init__(self, model: FrequencyDependentElectromagneticNeRF):
        self.model = model
        self.realtime_optimizer = RealTimeEMOptimizer(model)
        
    def calculate_path_loss(self, 
                          transmitter_pos: torch.Tensor,
                          receiver_pos: torch.Tensor,
                          frequency: torch.Tensor,
                          n_samples: int = 100) -> torch.Tensor:
        t = torch.linspace(0, 1, n_samples).unsqueeze(-1)
        path_points = (transmitter_pos.unsqueeze(0) * (1 - t) + 
                      receiver_pos.unsqueeze(0) * t)
        
        freq_expanded = frequency.expand(n_samples, 1)
        
        with torch.no_grad():
            predictions = self.model(path_points, freq_expanded)
            E_field = predictions['electric_field']
            
        power = torch.sum(E_field**2, dim=1)
        
        path_loss = -10 * torch.log10(torch.mean(power) + 1e-10)
        
        return path_loss
    
    def optimize_base_station_placement(self,
                                      coverage_area: torch.Tensor,
                                      frequency: torch.Tensor,
                                      n_candidates: int = 50) -> torch.Tensor:
        candidates = torch.rand(n_candidates, 3) * 2 - 1  # [-1, 1]^3
        
        best_loss = float('inf')
        best_position = None
        
        for candidate in candidates:
            total_loss = 0

            for coverage_point in coverage_area:
                path_loss = self.calculate_path_loss(
                    candidate.unsqueeze(0), 
                    coverage_point.unsqueeze(0), 
                    frequency
                )
                total_loss += path_loss
            
            if total_loss < best_loss:
                best_loss = total_loss
                best_position = candidate
                
    def optimize_mimo_beamforming(self, 
                                 base_station_pos: torch.Tensor,
                                 user_positions: torch.Tensor,
                                 frequency: torch.Tensor,
                                 n_antennas: int = 4) -> torch.Tensor:
        n_users = user_positions.shape[0]
        
        antenna_spacing = 3e8 / frequency.item() / 2
        antenna_positions = base_station_pos.unsqueeze(0) + torch.linspace(
            -antenna_spacing * (n_antennas-1) / 2,
            antenna_spacing * (n_antennas-1) / 2,
            n_antennas
        ).unsqueeze(-1) * torch.tensor([1., 0., 0.])
        
    
        channels = []
        for ant_pos in antenna_positions:
            user_channels = []
            for user_pos in user_positions:
                path_loss = self.calculate_path_loss(
                    ant_pos.unsqueeze(0), 
                    user_pos.unsqueeze(0), 
                    frequency
                )
                distance = torch.norm(ant_pos - user_pos)
                phase = 2 * math.pi * frequency * distance / 3e8
                
                channel = torch.exp(-path_loss/20) * torch.exp(1j * phase)
                user_channels.append(channel)
            
            channels.append(torch.stack(user_channels))
        
        channel_matrix = torch.stack(channels)
        
        H = channel_matrix.T
        H_conj_T = torch.conj(H).T
        
        try:
            beamforming_weights = H_conj_T @ torch.linalg.inv(H @ H_conj_T)
        except:
            reg_param = 1e-6
            beamforming_weights = H_conj_T @ torch.linalg.inv(
                H @ H_conj_T + reg_param * torch.eye(n_users)
            )
        
        return beamforming_weights
    
    def interference_analysis(self,
                            base_stations: torch.Tensor,
                            frequency: torch.Tensor,
                            analysis_grid: torch.Tensor) -> dict:
        results = {}
        
        signal_strengths = []
        for bs_pos in base_stations:
            strengths = []
            for grid_point in analysis_grid:
                path_loss = self.calculate_path_loss(
                    bs_pos.unsqueeze(0),
                    grid_point.unsqueeze(0), 
                    frequency
                )
                signal_strength = -path_loss
                strengths.append(signal_strength)
            signal_strengths.append(torch.stack(strengths))
        
        signal_matrix = torch.stack(signal_strengths)
        
        strongest_signal = torch.max(signal_matrix, dim=0)[0]
        interference = torch.sum(torch.exp(signal_matrix/10), dim=0) - torch.exp(strongest_signal/10)
        interference_db = 10 * torch.log10(interference + 1e-10)
        
        sinr = strongest_signal - interference_db
        
        results['signal_strength'] = strongest_signal
        results['interference'] = interference_db
        results['sinr'] = sinr
        results['coverage_probability'] = torch.mean((sinr > 10).float())
        
        return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = FrequencyDependentElectromagneticNeRF(
        hidden_dim=128,
        n_layers=6,
        encoding_config={
            'n_levels': 12,
            'n_features_per_level': 4,
            'base_resolution': 32,
            'finest_resolution': 1024
        }
    ).to(device)
    
    physics_loss = ImprovedMaxwellPhysicsLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    for epoch in range(100):
        optimizer.zero_grad()
        
        positions = torch.rand(1000, 3, device=device)
        frequency = torch.ones(1000, 1, device=device) * 2.4e9
        
        predictions = model(positions, frequency)
        
        physics_losses = physics_loss(predictions, positions)

        physics_losses['total_physics'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Physics Loss = {physics_losses['total_physics'].item():.6f}")
    
    optimizer_comm = CommunicationSystemOptimizer(model)
    
    coverage_area = torch.rand(10, 3) * 0.5
    frequency = torch.tensor([[2.4e9]])
    
    optimal_position = optimizer_comm.optimize_base_station_placement(
        coverage_area, frequency
    )
    
    print(f"최적 기지국 위치: {optimal_position}")
    print("훈련 및 최적화 완료!")

if __name__ == "__main__":
    main()