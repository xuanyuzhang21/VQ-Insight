# Install the packages in r1-v .
cd src/r1-v 
pip install -e ".[dev]"

# Addtional modules
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install flash-attn==2.7.4.post1 --no-cache-dir --no-build-isolation

# vLLM support 
pip install vllm==0.7.2

pip install trl==0.16.0

pip install nltk

pip install rouge_score

pip3 install deepspeed

pip3 install byted-wandb

cd ./src/qwen-vl-utils
pip install -e .[decord]

# fix transformers version
cd ./transformers-main
pip install -e .