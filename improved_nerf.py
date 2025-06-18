import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from PIL import Image
from tqdm import tqdm

class Positional_Encoding:
    
    def __init__(self, L):
        self.L = L

    def encode(self, p):
        
        encoding = [p]  
        for i in range(self.L):
            encoding.append(torch.sin(2**i * np.pi * p))
            encoding.append(torch.cos(2**i * np.pi * p))
        return torch.cat(encoding, dim=-1)

class NeRF_Network(nn.Module):
    
    def __init__(self, pos_L=10, dir_L=4, hidden_dim=256, use_viewdirs=True):
        super().__init__()
        
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
        self.feature_output = nn.Linear(hidden_dim, 128)  
        
        if use_viewdirs:
            self.color_layer = nn.Linear(128 + dir_input_dim, hidden_dim // 2)
            self.rgb_layer = nn.Linear(hidden_dim // 2, 3)
        else:
            self.color_layer = nn.Linear(128, hidden_dim // 2)
            self.rgb_layer = nn.Linear(hidden_dim // 2, 3)

    def forward(self, positions, directions=None):
        pos_encoded = self.pos_encoding.encode(positions)
        
        x = pos_encoded
        for i, layer in enumerate(self.density_layers):
            if i == 4:  
                x = torch.cat([x, pos_encoded], dim=-1)
            x = F.relu(layer(x))
        
        sigma = self.density_output(x) 
        features = self.feature_output(x)
        
        if self.use_viewdirs and directions is not None:
            directions = F.normalize(directions, p=2, dim=-1)
            dir_encoded = self.dir_encoding.encode(directions)
            color_input = torch.cat([features, dir_encoded], dim=-1)
        else:
            color_input = features
            
        color_features = F.relu(self.color_layer(color_input))
        rgb = torch.sigmoid(self.rgb_layer(color_features))
        
        return rgb, sigma

class VolumeRendering(nn.Module):
    def __init__(self):
        super().__init__()

    def stratified_sampling(self, rays_o, rays_d, near, far, N_samples, perturb=True):
        batch_size = rays_o.shape[0]
        
        t_vals = torch.linspace(0., 1., steps=N_samples, device=rays_o.device)
        z_vals = near * (1.-t_vals) + far * t_vals
        z_vals = z_vals.expand([batch_size, N_samples])
        
        if perturb:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            
            t_rand = torch.rand(z_vals.shape, device=z_vals.device)
            z_vals = lower + (upper - lower) * t_rand
        
        return z_vals
    
    def volume_render(self, sigma, colors, z_vals, rays_d, white_bkgd=False):
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)], dim=-1)
        
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        sigma = F.relu(sigma)
        
        alpha = 1.0 - torch.exp(-sigma * dists)
        
        transmittance = torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.-alpha + 1e-10], -1), 
            -1
        )[:, :-1]
        
        weights = transmittance * alpha
        
        rgb = torch.sum(weights[..., None] * colors, dim=-2)
        
        if white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb = rgb + (1. - acc_map[..., None])
        
        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)
        
        return rgb, weights, depth_map, disp_map, acc_map

class HierarchicalSampling(nn.Module):
    def __init__(self):
        super().__init__()

    def sample_pdf(self, bins, weights, N_samples, det=False):
        weights = weights + 1e-5 
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
        
        if det:
            u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)
        
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)
        
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
        
        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        
        return samples

class NeRF_Complete(nn.Module):
    
    def __init__(self, pos_L=10, dir_L=4, hidden_dim=256, use_viewdirs=True, use_ndc=False):
        super().__init__()
        
        self.coarse_network = NeRF_Network(pos_L, dir_L, hidden_dim, use_viewdirs)
        self.fine_network = NeRF_Network(pos_L, dir_L, hidden_dim, use_viewdirs)
        
        self.volume_renderer = VolumeRendering()
        self.hierarchical_sampler = HierarchicalSampling()
        
        self.use_ndc = use_ndc
        
    def ndc_rays(self, H, W, focal, near, rays_o, rays_d):
        t = -(near + rays_o[..., 2]) / rays_d[..., 2]
        rays_o = rays_o + t[..., None] * rays_d
        
        o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
        o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
        o2 = 1. + 2. * near / rays_o[..., 2]
        
        d0 = -1./(W/(2.*focal)) * (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
        d1 = -1./(H/(2.*focal)) * (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
        d2 = -2. * near / rays_o[..., 2]
        
        rays_o = torch.stack([o0, o1, o2], -1)
        rays_d = torch.stack([d0, d1, d2], -1)
        
        return rays_o, rays_d
        
    def render_rays(self, rays_o, rays_d, near, far, N_coarse=64, N_fine=128, 
                    perturb=True, white_bkgd=False, H=None, W=None, focal=None):
        
        if self.use_ndc and H is not None and W is not None and focal is not None:
            rays_o, rays_d = self.ndc_rays(H, W, focal, near, rays_o, rays_d)
            near, far = 0., 1.
        
        z_coarse = self.volume_renderer.stratified_sampling(
            rays_o, rays_d, near, far, N_coarse, perturb=perturb
        )
        
        pts_coarse = rays_o[..., None, :] + rays_d[..., None, :] * z_coarse[..., :, None]
        
        rays_d_expanded = rays_d[..., None, :].expand(pts_coarse.shape)
        
        rgb_coarse, sigma_coarse = self.coarse_network(
            pts_coarse.reshape(-1, 3), 
            rays_d_expanded.reshape(-1, 3)
        )
        
        rgb_coarse = rgb_coarse.reshape(rays_o.shape[0], N_coarse, 3)
        sigma_coarse = sigma_coarse.reshape(rays_o.shape[0], N_coarse)
        
        rgb_coarse_final, weights_coarse, depth_coarse, disp_coarse, acc_coarse = \
            self.volume_renderer.volume_render(
                sigma_coarse, rgb_coarse, z_coarse, rays_d, white_bkgd
            )
        
        z_vals_mid = .5 * (z_coarse[..., 1:] + z_coarse[..., :-1])
        z_fine = self.hierarchical_sampler.sample_pdf(
            z_vals_mid, weights_coarse[..., 1:-1], N_fine, det=(perturb==False)
        )
        z_fine = z_fine.detach()  
        
        z_combined, _ = torch.sort(torch.cat([z_coarse, z_fine], -1), -1)
        
        pts_fine = rays_o[..., None, :] + rays_d[..., None, :] * z_combined[..., :, None]
        
        rays_d_fine_expanded = rays_d[..., None, :].expand(pts_fine.shape)
        
        rgb_fine, sigma_fine = self.fine_network(
            pts_fine.reshape(-1, 3),
            rays_d_fine_expanded.reshape(-1, 3)
        )

        rgb_fine = rgb_fine.reshape(rays_o.shape[0], N_coarse + N_fine, 3)
        sigma_fine = sigma_fine.reshape(rays_o.shape[0], N_coarse + N_fine)
        
        rgb_fine_final, weights_fine, depth_fine, disp_fine, acc_fine = \
            self.volume_renderer.volume_render(
                sigma_fine, rgb_fine, z_combined, rays_d, white_bkgd
            )
        
        outputs = {
            'rgb_coarse': rgb_coarse_final,
            'rgb_fine': rgb_fine_final,
            'depth_coarse': depth_coarse,
            'depth_fine': depth_fine,
            'disp_coarse': disp_coarse,
            'disp_fine': disp_fine,
            'acc_coarse': acc_coarse,
            'acc_fine': acc_fine,
            'weights': weights_fine,
        }
        
        return outputs

def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W), 
        torch.linspace(0, H-1, H),
        indexing='xy'
    )
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

def sample_rays_batch(image, pose, focal, H, W, N_rays=1024):
    rays_o, rays_d = get_rays(H, W, focal, pose)
    
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    target_rgb = image.reshape(-1, 3)
    
    if N_rays < rays_o.shape[0]:
        indices = torch.randperm(rays_o.shape[0])[:N_rays]
        rays_o = rays_o[indices]
        rays_d = rays_d[indices]
        target_rgb = target_rgb[indices]
    
    return rays_o, rays_d, target_rgb

def compute_loss(outputs, target_rgb):
    loss_coarse = F.mse_loss(outputs['rgb_coarse'], target_rgb)
    loss_fine = F.mse_loss(outputs['rgb_fine'], target_rgb)
    return loss_coarse + loss_fine

def compute_psnr(mse):
    return -10. * torch.log10(mse)

class NeRFDataset(torch.utils.data.Dataset):
    def __init__(self, basedir, split='train', downsample=1.0, testskip=1):
        self.basedir = basedir
        self.split = split
        
        with open(os.path.join(basedir, f'transforms_{split}.json'), 'r') as f:
            self.meta = json.load(f)
        
        self.imgs = []
        self.poses = []
        
        skip = 1 if split != 'test' else testskip
        
        for idx, frame in enumerate(self.meta['frames']):
            if idx % skip != 0:
                continue
                
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            img = Image.open(fname)
            
            if downsample != 1.0:
                W, H = img.size
                img = img.resize((int(W/downsample), int(H/downsample)), Image.LANCZOS)
            
            img = np.array(img).astype(np.float32) / 255.0
            if img.shape[-1] == 4:
                img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])
            
            self.imgs.append(img)
            self.poses.append(np.array(frame['transform_matrix']).astype(np.float32))
        
        self.imgs = np.stack(self.imgs, 0)
        self.poses = np.stack(self.poses, 0)
        
        H, W = self.imgs[0].shape[:2]
        camera_angle_x = float(self.meta['camera_angle_x'])
        self.focal = .5 * W / np.tan(.5 * camera_angle_x)
        self.H, self.W = H, W
        
        self.near = 2.
        self.far = 6.
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        return {
            'image': torch.FloatTensor(self.imgs[idx]),
            'pose': torch.FloatTensor(self.poses[idx]),
        }

def train_nerf(model, train_dataset, val_dataset, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    def lr_lambda(iter):
        decay_rate = 0.1
        decay_steps = args.lr_decay * 1000
        return decay_rate ** (iter / decay_steps)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    model.train()
    
    pbar = tqdm(range(args.n_iters), desc="Training")
    for i in pbar:
        idx = np.random.randint(0, len(train_dataset))
        data = train_dataset[idx]
        
        rays_o, rays_d, target_rgb = sample_rays_batch(
            data['image'], data['pose'], 
            train_dataset.focal, train_dataset.H, train_dataset.W,
            N_rays=args.batch_size
        )
        
        rays_o = rays_o.to(args.device)
        rays_d = rays_d.to(args.device)
        target_rgb = target_rgb.to(args.device)
        
        outputs = model.render_rays(
            rays_o, rays_d, 
            train_dataset.near, train_dataset.far,
            N_coarse=args.N_coarse, N_fine=args.N_fine,
            perturb=True, white_bkgd=args.white_bkgd
        )
        
        loss = compute_loss(outputs, target_rgb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if i % 100 == 0:
            psnr = compute_psnr(F.mse_loss(outputs['rgb_fine'], target_rgb))
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'PSNR': f"{psnr.item():.2f}",
                'LR': f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        if i % args.val_freq == 0 and i > 0:
            validate(model, val_dataset, args)
            model.train()

        if i % args.save_freq == 0 and i > 0:
            torch.save({
                'iter': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.logdir, f'checkpoint_{i:06d}.pt'))

@torch.no_grad()
def validate(model, dataset, args):
    model.eval()
    
    idx = 0  
    data = dataset[idx]
    
    pose = data['pose'].to(args.device)
    H, W = dataset.H, dataset.W
    focal = dataset.focal
    
    rgb = render_image(model, pose, H, W, focal, dataset.near, dataset.far, args)
    
    target = data['image'].numpy()
    mse = np.mean((rgb - target) ** 2)
    psnr = -10. * np.log10(mse)
    
    print(f"\nValidation - PSNR: {psnr:.2f}")
    
    if args.logdir:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(rgb)
        axes[0].set_title(f'Rendered (PSNR: {psnr:.2f})')
        axes[0].axis('off')
        axes[1].imshow(target)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(args.logdir, f'val_{model.training_iter:06d}.png'))
        plt.close()

@torch.no_grad()
def render_image(model, pose, H, W, focal, near, far, args):
    model.eval()
    
    rays_o, rays_d = get_rays(H, W, focal, pose)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    chunk_size = args.chunk_size
    rgb_list = []
    
    for i in range(0, rays_o.shape[0], chunk_size):
        rays_o_chunk = rays_o[i:i+chunk_size].to(args.device)
        rays_d_chunk = rays_d[i:i+chunk_size].to(args.device)
        
        outputs = model.render_rays(
            rays_o_chunk, rays_d_chunk, near, far,
            N_coarse=args.N_coarse, N_fine=args.N_fine,
            perturb=False, white_bkgd=args.white_bkgd
        )
        rgb_list.append(outputs['rgb_fine'].cpu())
    
    rgb = torch.cat(rgb_list, 0)
    rgb = rgb.reshape(H, W, 3).numpy()
    
    return rgb

def test_improved_nerf():
    print("개선된 NeRF 테스트 시작...")
    
    model = NeRF_Complete(pos_L=10, dir_L=4, hidden_dim=256, use_viewdirs=True)
    
    batch_size = 1024
    rays_o = torch.randn(batch_size, 3)
    rays_d = torch.randn(batch_size, 3)
    rays_d = F.normalize(rays_d, p=2, dim=-1)
    
    near = 2.0
    far = 6.0
    
    with torch.no_grad():
        outputs = model.render_rays(rays_o, rays_d, near, far)
    
    print(f"입력 광선 개수: {batch_size}")
    print(f"Coarse RGB 출력: {outputs['rgb_coarse'].shape}")
    print(f"Fine RGB 출력: {outputs['rgb_fine'].shape}")
    print(f"Depth map shape: {outputs['depth_fine'].shape}")
    print("테스트 완료!")

if __name__ == "__main__":
    test_improved_nerf()