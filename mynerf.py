import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms



class Positional_Encoding:
    
    def __init__(self, L):
        self.L = L

    def encode(self, p):
        encoding = []   
    
        for i in range(self.L):
            encoding.append(torch.sin(2**i * np.pi * p))
            encoding.append(torch.cos(2**i * np.pi * p))

        return torch.cat(encoding, dim=-1)


class VolumeRendering(nn.Module):
    """
    원래 NeRF의 Volume Rendering 구현
    Paper: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
    """

    def __init__(self):
        super().__init__()

    def stratified_sampling(self, rays_o, rays_d, near, far, N_samples, perturb=True):
        """
        Stratified sampling along rays
        
        Args:
            rays_o: [batch_size, 3] ray origins
            rays_d: [batch_size, 3] ray directions  
            near, far: near and far plane distances
            N_samples: number of samples per ray
            perturb: whether to add random perturbation (training시 True)
        Returns:
            z_vals: [batch_size, N_samples] sample distances along rays
        """
        # 균등하게 N_samples개 구간으로 나누기
        t_vals = torch.linspace(0., 1., steps=N_samples, device=rays_o.device)
        z_vals = near * (1.0 - t_vals) + far * t_vals
        z_vals = z_vals.expand([rays_o.shape[0], N_samples])
        
        # Training 중에는 각 구간 내에서 랜덤하게 샘플링 (Stratified Sampling)
        if perturb:
            # 각 구간의 중점들 계산
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            
            # 각 구간 내에서 균등분포로 랜덤 샘플링
            t_rand = torch.rand(z_vals.shape, device=z_vals.device)
            z_vals = lower + (upper - lower) * t_rand
        
        return z_vals
    
    def volume_render(self, rgb, sigma, z_vals, rays_d, white_bkgd=False, noise_std=0.0):
        """
        Original NeRF Volume Rendering Equation
        
        C(r) = ∫ T(t) * σ(t) * c(t) dt
        where T(t) = exp(-∫₀ᵗ σ(s) ds)
        
        Args:
            rgb: [batch_size, N_samples, 3] RGB colors for each sample
            sigma: [batch_size, N_samples] density values  
            z_vals: [batch_size, N_samples] sample distances along rays
            rays_d: [batch_size, 3] ray directions
            white_bkgd: whether to use white background
            noise_std: noise standard deviation for regularization
        Returns:
            rgb_map: [batch_size, 3] rendered RGB color
            depth_map: [batch_size] rendered depth
            acc_map: [batch_size] accumulated opacity
            weights: [batch_size, N_samples] weights for each sample
        """
        # 1. 인접한 샘플 간의 거리 계산 (δᵢ)
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        # 마지막 구간은 무한대로 설정
        dists = torch.cat([dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)], -1)
        
        # 실제 3D 거리로 변환 (ray direction의 norm 고려)
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        
        # 2. Training 중 노이즈 추가 (regularization)
        if noise_std > 0.0:
            noise = torch.randn_like(sigma) * noise_std
            sigma = sigma + noise
        
        # 3. 불투명도 계산: αᵢ = 1 - exp(-σᵢ * δᵢ)
        alpha = 1.0 - torch.exp(-F.relu(sigma) * dists)  # ReLU로 음수 밀도 방지
        
        # 4. 투과율 계산: Tᵢ = exp(-∑ⱼ₌₁ⁱ⁻¹ σⱼ * δⱼ) = ∏ⱼ₌₁ⁱ⁻¹ (1 - αⱼ)
        transparency = 1.0 - alpha + 1e-10  # 수치 안정성을 위한 epsilon
        transmittance = torch.cumprod(transparency, -1)
        # 첫 번째 샘플의 투과율은 1 (아무것도 지나지 않음)
        transmittance = torch.cat([torch.ones_like(transmittance[..., :1]), transmittance[..., :-1]], -1)
        
        # 5. 가중치 계산: wᵢ = Tᵢ * αᵢ  
        weights = transmittance * alpha
        
        # 6. 최종 색상 계산: C(r) = ∑ᵢ wᵢ * cᵢ
        rgb_map = torch.sum(weights[..., None] * rgb, -2)
        
        # 7. Depth map 계산: 가중 평균 깊이
        depth_map = torch.sum(weights * z_vals, -1)
        
        # 8. Accumulated opacity 계산 (전체 불투명도)
        acc_map = torch.sum(weights, -1)
        
        # 9. White background 처리 (synthetic data에서 사용)
        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])
        
        return rgb_map, depth_map, acc_map, weights

    def get_rays_3d_points(self, rays_o, rays_d, z_vals):
        """
        Ray 상의 3D 점들 계산
        
        Args:
            rays_o: [batch_size, 3] ray origins
            rays_d: [batch_size, 3] ray directions
            z_vals: [batch_size, N_samples] sample distances
        Returns:
            pts: [batch_size, N_samples, 3] 3D points along rays
        """
        # p(t) = o + t * d
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        return pts



class Hierarchical_sampling(nn.Module, VolumeRendering):

    def __init__(self):
        super().__init__()

    def weights_to_pdf(self, weights, z_vals):

        weight = weights + 1e-5
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)

        z_bins = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])

        return pdf, z_bins

    def sample_pdf(self, z_bins, pdf, N_fine):
        
        batch_size = pdf.shape[0]
        
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        
        u = torch.rand(batch_size, N_fine)
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

    def coarse_and_fine_sampling(self, z_coarse, weights_coarse, N_fine):
        
        pdf, z_bins = self.weights_to_pdf(weights_coarse, z_coarse)
        
        z_fine = self.sample_pdf(z_bins, pdf, N_fine)
        
        z_combined = torch.cat([z_coarse, z_fine], dim=-1)
        z_combined, _ = torch.sort(z_combined, dim=-1)
        
        return z_combined
    
    def coarse_network(self, pts_coarse, rays_d):
        rgb_coarse, sigma_coarse = (pts_coarse, rays_d)
        rgb_coarse = self.compute_final_color()
        return rgb_coarse, sigma_coarse
    
    def fine_network(self, pts_fine, rays_d):
        rgb_fine, sigma_fine = (pts_fine, rays_d)
        return rgb_fine, sigma_fine
    
    def render_rays_hierarchical(self, rays_o, rays_d, near, far):
        
        z_coarse = self.Stratified_sampling(rays_o, rays_d, near, far, N_coarse=64)
        pts_coarse = rays_o[..., None, :] + rays_d[..., None, :] * z_coarse[..., :, None]
        rgb_coarse, sigma_coarse = (pts_coarse, rays_d)
        rgb_coarse_final, weights_coarse = self.volume_render(sigma_coarse, rgb_coarse, z_coarse)
        
        z_fine = self.coarse_and_fine_sampling(z_coarse, weights_coarse, N_fine=128)
        
        pts_fine = rays_o[..., None, :] + rays_d[..., None, :] * z_fine[..., :, None]
        
        rgb_fine, sigma_fine = fine_network(pts_fine, rays_d)
        
        rgb_fine_final, weights_fine = self.volume_render(sigma_fine, rgb_fine, z_fine)
        
        return rgb_coarse_final, rgb_fine_final


class NeRF(nn.Module):
    
    def __init__(self, pos_L=10, dir_L=4, hidden_dim=256, use_viewdirs=True):
        super(NeRF, self).__init__()
        
        self.pos_encoding = Positional_Encoding(pos_L)
        self.dir_encoding = Positional_Encoding(dir_L)
        self.use_viewdirs = use_viewdirs
        
        pos_input_dim = 3 + 3 * 2 * pos_L 
        dir_input_dim = 3 + 3 * 2 * dir_L if use_viewdirs else 0

        self.density_layers = nn.ModuleList([
            nn.Linear(pos_input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim + pos_input_dim, hidden_dim),  
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])
        self.density_output = nn.Linear(hidden_dim, 1)
        self.feature_output = nn.Linear(hidden_dim, hidden_dim)
        
        if use_viewdirs:
            self.color_layer = nn.Linear(hidden_dim + dir_input_dim, hidden_dim // 2) 
            self.rgb_layer = nn.Linear(hidden_dim // 2, 3)
        else:
            self.color_layer = nn.Linear(hidden_dim, hidden_dim // 2)  
            self.rgb_layer = nn.Linear(hidden_dim // 2, 3)

    def forward(self, positions, directions=None):
        
        pos_encoded = self.pos_encoding.encode(positions)
        pos_input = torch.cat([positions, pos_encoded], dim=-1)
        x = pos_input
        for i, layer in enumerate(self.density_layers):
            if i == 4:  
                x = torch.cat([x, pos_input], dim=-1)
            x = F.relu(layer(x))
        
        sigma = F.relu(self.density_output(x))  
        features = self.feature_output(x)
        
        if self.use_viewdirs and directions is not None:
            dir_encoded = self.dir_encoding.encode(directions)
            dir_input = torch.cat([directions, dir_encoded], dim=-1)
            color_input = torch.cat([features, dir_input], dim=-1)
        else:
            color_input = features
            
        color_features = F.relu(self.color_layer(color_input))
        rgb = torch.sigmoid(self.rgb_layer(color_features))
        
        return rgb, sigma




class NDC(nn.Module):


    def __init__(self, H, W, focal):

        super(NDC, self).__init__()
        self.H = H
        self.W = W
        self.focal = focal

    def get_ndc_rays(self, rays_o, rays_d):

        t = -(1.0 + rays_o[..., 2:3]) / rays_d[..., 2:3]
        rays_o = rays_o + t * rays_d

        o0 = -1.0 / (self.W / (2.0 * self.focal)) * rays_o[..., 0:1] / rays_o[..., 2:3]
        o1 = -1.0 / (self.H / (2.0 * self.focal)) * rays_o[..., 1:2] / rays_o[..., 2:3]
        o2 = 1.0 + 2.0 * 1.0 / rays_o[..., 2:3]
        
        d0 = -1.0 / (self.W / (2.0 * self.focal)) * (rays_d[..., 0:1] / rays_d[..., 2:3] - rays_o[..., 0:1] / rays_o[..., 2:3])
        d1 = -1.0 / (self.H / (2.0 * self.focal)) * (rays_d[..., 1:2] / rays_d[..., 2:3] - rays_o[..., 1:2] / rays_o[..., 2:3])
        d2 = -2.0 * 1.0 / rays_o[..., 2:3]
        
        rays_o_ndc = torch.cat([o0, o1, o2], dim=-1)
        rays_d_ndc = torch.cat([d0, d1, d2], dim=-1)
        
        return rays_o_ndc, rays_d_ndc

