
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
tar -C /usr -xzf ollama-linux-amd64.tgz
//.bashrc
export PATH="/userhome/cs2/u3592844/bin:$PATH"
// ollama放进bin/
ollama serve & ollama run qwen2.5:14b



# RUN:
gpu-interactive
nohup ollama serve > ./output.log 2>&1 & 
conda 打开虚拟环境
python main_function.py 