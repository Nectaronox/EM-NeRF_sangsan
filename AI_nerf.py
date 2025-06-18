import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Positional_Encoding:
    def __init__(self, L):
        self.L = L

    def encode(self, p):
        encoding = []   
        for i in range(self.L):
            encoding.append(torch.sin(2**i * np.pi * p))
            encoding.append(torch.cos(2**i * np.pi * p))
        return torch.cat(encoding, dim=-1)

class NeRF_Network(nn.Module):
    """단일 NeRF 네트워크 (coarse와 fine에서 공통 사용)"""
    
    def __init__(self, pos_L=10, dir_L=4, hidden_dim=256, use_viewdirs=True):
        super().__init__()
        
        self.pos_encoding = Positional_Encoding(pos_L)
        self.dir_encoding = Positional_Encoding(dir_L)
        self.use_viewdirs = use_viewdirs
        
        # 입력 차원 계산
        pos_input_dim = 3 + 3 * 2 * pos_L  # 원본 좌표 + positional encoding
        dir_input_dim = 3 + 3 * 2 * dir_L if use_viewdirs else 0

        # 첫 번째 부분: 위치 -> 밀도 + 특징 (8-layer MLP with skip connection)
        self.density_layers = nn.ModuleList([
            nn.Linear(pos_input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim + pos_input_dim, hidden_dim),  # skip connection
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])
        self.density_output = nn.Linear(hidden_dim, 1)
        self.feature_output = nn.Linear(hidden_dim, hidden_dim)
        
        # 두 번째 부분: 특징 + 방향 -> RGB
        if use_viewdirs:
            self.color_layer = nn.Linear(hidden_dim + dir_input_dim, hidden_dim // 2)
            self.rgb_layer = nn.Linear(hidden_dim // 2, 3)
        else:
            self.color_layer = nn.Linear(hidden_dim, hidden_dim // 2)
            self.rgb_layer = nn.Linear(hidden_dim // 2, 3)

    def forward(self, positions, directions=None):
        # 위치 인코딩
        pos_encoded = self.pos_encoding.encode(positions)
        pos_input = torch.cat([positions, pos_encoded], dim=-1)
        
        # 8-layer MLP with skip connection
        x = pos_input
        for i, layer in enumerate(self.density_layers):
            if i == 4:  # skip connection at 5th layer
                x = torch.cat([x, pos_input], dim=-1)
            x = F.relu(layer(x))
        
        # 밀도와 특징 계산
        sigma = F.relu(self.density_output(x))  # 밀도는 양수
        features = self.feature_output(x)
        
        # RGB 계산
        if self.use_viewdirs and directions is not None:
            dir_encoded = self.dir_encoding.encode(directions)
            dir_input = torch.cat([directions, dir_encoded], dim=-1)
            color_input = torch.cat([features, dir_input], dim=-1)
        else:
            color_input = features
            
        color_features = F.relu(self.color_layer(color_input))
        rgb = torch.sigmoid(self.rgb_layer(color_features))
        
        return rgb, sigma

class VolumeRendering(nn.Module):
    def __init__(self):
        super().__init__()

    def stratified_sampling(self, rays_o, rays_d, near, far, N_samples):
        """올바른 Stratified Sampling 구현"""
        # 구간을 N개로 나누기
        t_vals = torch.linspace(0., 1., steps=N_samples+1, device=rays_o.device)
        lower = t_vals[:-1]  # [0, 1/N, 2/N, ..., (N-1)/N]
        upper = t_vals[1:]   # [1/N, 2/N, 3/N, ..., 1]
        
        # 각 구간 내에서 랜덤 샘플링
        u = torch.rand(rays_o.shape[0], N_samples, device=rays_o.device)
        t_vals = lower + (upper - lower) * u
        
        # 실제 거리값으로 변환
        z_vals = near * (1.0 - t_vals) + far * t_vals
        return z_vals
    
    def volume_render(self, sigma, colors, z_vals):
        """Volume Rendering 구현"""
        # 구간 거리 계산
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)], dim=-1)

        # 불투명도 계산
        alpha = 1.0 - torch.exp(-sigma * dists)
        
        # 투과율 계산 (누적곱)
        transparency = 1.0 - alpha + 1e-10
        transmittance = torch.cumprod(transparency, dim=-1)
        transmittance = torch.cat([torch.ones_like(transmittance[..., :1]), 
                                  transmittance[..., :-1]], dim=-1)
        
        # 가중치 계산
        weights = transmittance * alpha
        
        # 최종 색상 계산
        rgb = torch.sum(weights[..., None] * colors, dim=-2)
        
        return rgb, weights

class HierarchicalSampling(nn.Module):
    def __init__(self):
        super().__init__()

    def weights_to_pdf(self, weights, z_vals):
        """가중치를 확률밀도함수로 변환"""
        weights = weights + 1e-5  # 수치 안정성
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        z_bins = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        return pdf, z_bins

    def sample_pdf(self, z_bins, pdf, N_fine):
        """PDF를 바탕으로 새로운 샘플점들을 생성"""
        batch_size = pdf.shape[0]
        
        # 누적분포함수 계산
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        
        # 균등분포에서 랜덤 샘플링
        u = torch.rand(batch_size, N_fine, device=pdf.device)
        
        # 역변환 샘플링
        indices = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(indices - 1, 0, cdf.shape[-1] - 1)
        above = torch.clamp(indices, 0, cdf.shape[-1] - 1)
        
        indices_g = torch.stack([below, above], dim=-1)
        cdf_g = torch.gather(cdf[..., None], -2, indices_g)
        z_bins_g = torch.gather(z_bins[..., None], -2, indices_g)
        
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        
        z_samples = z_bins_g[..., 0] + t * (z_bins_g[..., 1] - z_bins_g[..., 0])
        return z_samples

    def hierarchical_sampling(self, z_coarse, weights_coarse, N_fine):
        """Hierarchical sampling 전체 과정"""
        pdf, z_bins = self.weights_to_pdf(weights_coarse, z_coarse)
        z_fine = self.sample_pdf(z_bins, pdf, N_fine)
        
        # Coarse + Fine 결합하고 정렬
        z_combined = torch.cat([z_coarse, z_fine], dim=-1)
        z_combined, _ = torch.sort(z_combined, dim=-1)
        return z_combined

class NeRF_Complete(nn.Module):
    """완전한 NeRF 구현 (Coarse + Fine Network)"""
    
    def __init__(self, pos_L=10, dir_L=4, hidden_dim=256, use_viewdirs=True):
        super().__init__()
        
        # 두 개의 독립적인 네트워크
        self.coarse_network = NeRF_Network(pos_L, dir_L, hidden_dim, use_viewdirs)
        self.fine_network = NeRF_Network(pos_L, dir_L, hidden_dim, use_viewdirs)
        
        # 렌더링 모듈들
        self.volume_renderer = VolumeRendering()
        self.hierarchical_sampler = HierarchicalSampling()
        
    def render_rays(self, rays_o, rays_d, near, far, N_coarse=64, N_fine=128):
        """
        광선 렌더링 전체 파이프라인
        
        Args:
            rays_o: (batch_size, 3) 광선 시작점
            rays_d: (batch_size, 3) 광선 방향
            near, far: near/far plane 거리
            N_coarse: coarse 샘플 개수
            N_fine: fine 샘플 개수
            
        Returns:
            rgb_coarse: coarse 네트워크 결과
            rgb_fine: fine 네트워크 결과 (최종 출력)
        """
        
        # === COARSE PASS ===
        # 1. Coarse 샘플링
        z_coarse = self.volume_renderer.stratified_sampling(
            rays_o, rays_d, near, far, N_coarse
        )
        
        # 2. 3D 점 계산
        pts_coarse = rays_o[..., None, :] + rays_d[..., None, :] * z_coarse[..., :, None]
        
        # 3. 방향 벡터 확장 (모든 샘플점에 동일한 방향)
        rays_d_expanded = rays_d[..., None, :].expand(pts_coarse.shape)
        
        # 4. Coarse network 실행
        rgb_coarse, sigma_coarse = self.coarse_network(
            pts_coarse.reshape(-1, 3), 
            rays_d_expanded.reshape(-1, 3)
        )
        
        # 5. 원래 shape로 복원
        rgb_coarse = rgb_coarse.reshape(rays_o.shape[0], N_coarse, 3)
        sigma_coarse = sigma_coarse.reshape(rays_o.shape[0], N_coarse)
        
        # 6. Coarse 렌더링
        rgb_coarse_final, weights_coarse = self.volume_renderer.volume_render(
            sigma_coarse, rgb_coarse, z_coarse
        )
        
        # === FINE PASS ===
        # 7. Hierarchical 샘플링
        z_fine = self.hierarchical_sampler.hierarchical_sampling(
            z_coarse, weights_coarse, N_fine
        )
        
        # 8. 3D 점 계산
        pts_fine = rays_o[..., None, :] + rays_d[..., None, :] * z_fine[..., :, None]
        
        # 9. 방향 벡터 확장
        rays_d_fine_expanded = rays_d[..., None, :].expand(pts_fine.shape)
        
        # 10. Fine network 실행
        rgb_fine, sigma_fine = self.fine_network(
            pts_fine.reshape(-1, 3),
            rays_d_fine_expanded.reshape(-1, 3)
        )
        
        # 11. 원래 shape로 복원
        rgb_fine = rgb_fine.reshape(rays_o.shape[0], N_coarse + N_fine, 3)
        sigma_fine = sigma_fine.reshape(rays_o.shape[0], N_coarse + N_fine)
        
        # 12. Fine 렌더링 (최종 결과)
        rgb_fine_final, weights_fine = self.volume_renderer.volume_render(
            sigma_fine, rgb_fine, z_fine
        )
        
        return rgb_coarse_final, rgb_fine_final

# 테스트 함수
def test_complete_nerf():
    """완전한 NeRF 테스트"""
    print("완전한 NeRF 테스트 시작...")
    
    # 모델 초기화
    model = NeRF_Complete(pos_L=10, dir_L=4, hidden_dim=256, use_viewdirs=True)
    
    # 더미 데이터 생성
    batch_size = 1024
    rays_o = torch.randn(batch_size, 3)
    rays_d = torch.randn(batch_size, 3)
    rays_d = F.normalize(rays_d, p=2, dim=-1)  # 방향벡터 정규화
    
    near = 2.0
    far = 6.0
    
    # Forward pass
    with torch.no_grad():
        rgb_coarse, rgb_fine = model.render_rays(rays_o, rays_d, near, far)
    
    print(f"입력 광선 개수: {batch_size}")
    print(f"Coarse RGB 출력: {rgb_coarse.shape}")
    print(f"Fine RGB 출력: {rgb_fine.shape}")
    print("테스트 완료!")

if __name__ == "__main__":
    test_complete_nerf()