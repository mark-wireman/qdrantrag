
#useradd -r -s /bin/false -m -d /usr/share/ollama ollama
#systemctl daemon-reload
#systemctl enable ollama
#systemctl start ollama
ollama serve &>/dev/null &
ollama pull mistral
mount -t cifs -o username='mwireman',password='shinobu5' //192.168.50.121/sambashare/Literature /mnt/docs
mount -t cifs -o username='mwireman',password='shinobu5' //192.168.50.121/sambashare/qdrant/ollama /mnt/ollama
curl localhost:11434
#/bin/bash
python3 /home/app/vectordbgen.py

