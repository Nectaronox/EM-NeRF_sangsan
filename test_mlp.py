import torch
import torch.nn as nn
import numpy as np

def test_linear_vs_relu():
    """nn.Linear와 nn.ReLU의 차이점을 보여주는 테스트"""
    print("🔍 nn.Linear vs nn.ReLU 비교 테스트")
    print("="*50)
    
    # 입력 데이터
    batch_size = 4
    input_dim = 3
    x = torch.randn(batch_size, input_dim)
    print(f"입력 데이터 크기: {x.shape}")
    print(f"입력 데이터:\n{x}")
    print()
    
    # 1. nn.Linear만 사용
    linear = nn.Linear(input_dim, 5)
    output_linear = linear(x)
    print(f"🟦 nn.Linear(3→5) 결과:")
    print(f"출력 크기: {output_linear.shape}")
    print(f"출력 데이터:\n{output_linear}")
    print(f"학습 가능한 파라미터 수: {sum(p.numel() for p in linear.parameters())}")
    print()
    
    # 2. nn.ReLU만 사용 (의미 없음)
    relu = nn.ReLU()
    output_relu = relu(x)  # 입력과 동일한 크기
    print(f"🟨 nn.ReLU() 결과:")
    print(f"출력 크기: {output_relu.shape}")
    print(f"출력 데이터:\n{output_relu}")
    print(f"학습 가능한 파라미터 수: {sum(p.numel() for p in relu.parameters() if p.requires_grad)}")
    print()
    
    # 3. 올바른 조합: nn.Linear + nn.ReLU
    mlp = nn.Sequential(
        nn.Linear(input_dim, 5),
        nn.ReLU()
    )
    output_mlp = mlp(x)
    print(f"✅ nn.Linear + nn.ReLU 결과:")
    print(f"출력 크기: {output_mlp.shape}")
    print(f"출력 데이터:\n{output_mlp}")
    print(f"학습 가능한 파라미터 수: {sum(p.numel() for p in mlp.parameters())}")


def analyze_nerf_mlp():
    """NeRF MLP의 구조와 파라미터 수 분석"""
    print("\n" + "="*50)
    print("🧠 NeRF MLP 구조 분석")
    print("="*50)
    
    # NeRF 설정
    pos_L = 10
    dir_L = 4
    hidden_dim = 256
    
    # 입력 차원 계산
    pos_input_dim = 3 + 3 * 2 * pos_L  # 63
    dir_input_dim = 3 + 3 * 2 * dir_L  # 27
    
    print(f"📍 위치 입력 차원: {pos_input_dim}")
    print(f"📍 방향 입력 차원: {dir_input_dim}")
    print()
    
    # Density Network 분석
    density_net = nn.Sequential(
        nn.Linear(pos_input_dim, hidden_dim),  # 63 → 256
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),     # 256 → 256
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),     # 256 → 256
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),     # 256 → 256
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim + 1)  # 256 → 257
    )
    
    # Color Network 분석
    color_net = nn.Sequential(
        nn.Linear(hidden_dim + dir_input_dim, hidden_dim // 2),  # 283 → 128
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, 3),  # 128 → 3
        nn.Sigmoid()
    )
    
    # 파라미터 수 계산
    density_params = sum(p.numel() for p in density_net.parameters())
    color_params = sum(p.numel() for p in color_net.parameters())
    total_params = density_params + color_params
    
    print(f"🏗️ Density Network:")
    print(f"   - 레이어 수: 5개")
    print(f"   - 파라미터 수: {density_params:,}개")
    print()
    
    print(f"🎨 Color Network:")
    print(f"   - 레이어 수: 2개")
    print(f"   - 파라미터 수: {color_params:,}개")
    print()
    
    print(f"📊 전체 NeRF 네트워크:")
    print(f"   - 총 파라미터 수: {total_params:,}개")
    print(f"   - 메모리 사용량 (float32): {total_params * 4 / 1024 / 1024:.2f} MB")


def demonstrate_mlp_layers():
    """MLP의 각 레이어가 하는 일을 시각화"""
    print("\n" + "="*50)
    print("🔬 MLP 레이어별 변환 과정")
    print("="*50)
    
    # 간단한 3층 MLP
    mlp = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )
    
    # 입력 데이터
    x = torch.tensor([[1.0, 2.0, 3.0]])
    print(f"입력: {x} (크기: {x.shape})")
    
    # 단계별 변환 추적
    current = x
    for i, layer in enumerate(mlp):
        current = layer(current)
        layer_name = type(layer).__name__
        print(f"레이어 {i+1} ({layer_name}): {current.detach().numpy().flatten()[:3]}... (크기: {current.shape})")


if __name__ == "__main__":
    test_linear_vs_relu()
    analyze_nerf_mlp()
    demonstrate_mlp_layers() 