#!/bin/bash

echo "开始创建并配置环境..."

# 确保conda已初始化
if ! grep -q "conda init" ~/.bashrc; then
    echo "正在初始化conda..."
    conda init bash
    source ~/.bashrc
fi

# 创建conda环境并自动回答y
echo "正在创建conda环境e0..."
conda create -n t710 -y
if [ $? -ne 0 ]; then
    echo "创建conda环境失败，终止执行"
    exit 1
fi

# 激活环境
echo "正在激活环境e0..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate t710
if [ $? -ne 0 ]; then
    echo "激活环境失败，终止执行"
    exit 1
fi

# 安装PyTorch并自动回答y
echo "正在安装PyTorch..."
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y
if [ $? -ne 0 ]; then
    echo "PyTorch安装失败，终止执行"
    exit 1
fi

# 更新环境并自动回答y
echo "正在更新环境..."
conda env update --file environment.yml
if [ $? -ne 0 ]; then
    echo "环境更新失败，终止执行"
    exit 1
fi

# 安装diff-gaussian-rasterization
echo "正在安装diff-gaussian-rasterization..."
cd submodules/diff-gaussian-rasterization
pip install -e .
cd ..
if [ $? -ne 0 ]; then
    echo "diff-gaussian-rasterization安装失败，终止执行"
    exit 1
fi

# 安装simple-knn
echo "正在安装simple-knn..."
cd simple-knn
pip install -e .
cd ..
cd ..
if [ $? -ne 0 ]; then
    echo "simple-knn安装失败，终止执行"
    exit 1
fi

echo "渲染"
#python train.py --eval -s /LiuHan/sort-free-gs/data/train -m /LiuHan/sort-free-gs/output/test3 --depth_correct
python render.py -m /LiuHan/sort-free-gs/output/test1 --store_image
#python render.py -m /LiuHan/mini-splatting/output/baseline --store_image
echo "所有命令执行完成！"
