echo "=== Python 환경 설정 ==="

# 가상환경 생성
python -m venv nerf_env
source nerf_env/bin/activate

# 필수 패키지 설치
pip install torch torchvision
pip install numpy matplotlib tqdm
pip install imageio imageio-ffmpeg
pip install Pillow

echo "Environment setup complete!"


