#!/bin/bash
OLLAMA_HOST=0.0.0.0
export OLLAMA_HOST
PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
export PATH
LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export LD_LIBRARY_PATH
NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NVIDIA_DRIVER_CAPABILITIES

echo "Mounting the samambashare drive locations.\n"
mount -t cifs -o username='mwireman',password='shinobu5' //192.168.50.121/sambashare/Literature /mnt/docs
mount -t cifs -o username='mwireman',password='shinobu5' //192.168.50.121/sambashare/qdrant/ollama /mnt/ollama

#ollama serve &>/dev/null &
nohup ollama serve > /dev/null 2>&1 &
ps -ef | grep "ollama" | grep -v grep
#ollama pull llama2-uncensored
ollama pull mixtral

python3 /home/app/private_chat.py