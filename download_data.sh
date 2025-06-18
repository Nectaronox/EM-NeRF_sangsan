echo "=== NeRF 데이터셋 다운로드 스크립트 ==="
mkdir -p data
cd data

# Synthetic NeRF 데이터셋 다운로드
echo "Downloading Synthetic NeRF dataset..."
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_synthetic.zip
unzip nerf_synthetic.zip
rm nerf_synthetic.zip

# LLFF 데이터셋 다운로드 (선택사항)
# echo "Downloading LLFF dataset..."
# wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_llff_data.zip
# unzip nerf_llff_data.zip
# rm nerf_llff_data.zip

cd ..
echo "Dataset download complete!"
