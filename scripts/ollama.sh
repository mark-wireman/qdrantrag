#!/bin/bash


mount -t cifs -o username='mwireman',password='shinobu5' //192.168.50.121/sambashare/data/vectordb /mnt/data
mount -t cifs -o username='mwireman',password='shinobu5' //192.168.50.121/sambashare/Literature /mnt/docs
mount -t cifs -o username='mwireman',password='shinobu5' //192.168.50.121/sambashare/qdrant/ollama /mnt/ollama

#nohup ollama serve > /dev/null 2>&1 &
ollama serve
#ps -ef | grep "ollama" | grep -v grep