# authors note known issues with 11.8
pip install "torch<2.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
pip install plyfile;
pip install tqdm;
pip install submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim opencv-python joblib;
pip install "numpy<2.0";
pip install gnureadline;

# in case problems with cuda
# pip uninstall torch torchvision torchaudio -y
# pip cache purge
# rm -rf ~/.cache/pip ~/.local/lib/python3.10/site-packages/torch
# rm -rf ~/gaussian-splatting/.venv/lib/python3.10/site-packages/torch
# deactivate
# source .venv/bin/activate

# if having problems with submodule
# pip uninstall simple-knn -y
# rm -rf ~/gaussian-splatting/.venv/lib/python3.10/site-packages/simple_knn*
# rm -rf ~/gaussian-splatting/submodules/simple-knn/build
# rm -rf ~/gaussian-splatting/submodules/simple-knn/*.so

# if having problems with fused-ssim
# pip uninstall fused-ssim -y
# rm -rf ~/gaussian-splatting/.venv/lib/python3.10/site-packages/fused_ssim*
# rm -rf ~/gaussian-splatting/submodules/fused-ssim/build
# rm -rf ~/gaussian-splatting/submodules/fused-ssim/*.so