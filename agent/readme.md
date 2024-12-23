
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
tar -C /usr -xzf ollama-linux-amd64.tgz

export PATH="/userhome/cs2/u3592844/bin:$PATH"
ollama serve & ollama run qwen2.5:14b

