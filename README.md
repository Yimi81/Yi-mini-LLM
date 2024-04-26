# Yi-mini-LLM
Created by Yimi81

## 📝 介绍
本项目旨在从0到1构建一个小参数量的中文大模型，在实践的过程中学习预训练/SFT数据处理，模型架构，tokenizer, 参数设置，分布式训练与监控，对齐，推理部署等LLM核心知识。

## 🏋️‍️ 快速开始

### 环境准备
```python
git clone https://github.com/Yimi81/Yi-mini-LLM.git
cd Yi-mini-LLM
conda create -n mini-llm python=3.10 -y
conda activate mini-llm
pip install -r requirements.txt
pip install deepspeed
pip install flash-attn --no-build-isolation
```

### 数据集下载

#### 预训练数据集
```python
# 以天工开源的预训练数据集为例，太大了所以下载一部分
mkdir data; cd data; mkdir skypile; cd skypile
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/Skywork/SkyPile-150B
cd SkyPile-150B/data
git lfs pull --include "2023*.jsonl"
```

### 预训练
```python
bash scripts/pretrain.sh
```

### SFT (Readme待完善)
```python
bash scripts/sft.sh
```

### 推理
```python
python inference.py --model "your-pretrain-model-path"
```

## 🎓 参考
1. https://github.com/DLLXW/baby-llama2-chinese

2. https://github.com/jiahe7ay/MINI_LLM

3. https://github.com/yangjianxin1/Firefly

4. https://github.com/hiyouga/LLaMA-Factory


