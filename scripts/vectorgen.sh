#!/bin/bash

MODELNAME=$1
CHUNKSIZE=$2
CHUNKOVERLAP=$3
OLLAMAURL=$4
OLLAMAPORTNO=$5
PERSISTDIR=$6
SOURCEDIR=$7

echo "Model name = $MODELNAME"
echo "Ollama URL = $OLLAMAURL"

echo "Mounting the samambashare drive locations."
mount -t cifs -o username='mwireman',password='shinobu5' //192.168.50.121/sambashare/Literature /mnt/docs
mount -t cifs -o username='mwireman',password='shinobu5' //192.168.50.121/sambashare/qdrant/ollama /mnt/ollama
mount -t cifs -o username='mwireman',password='shinobu5' //192.168.50.121/sambashare/data/vectordb /mnt/data

echo "Pulling the mistral model."
CURLURL="$OLLAMAURL:$OLLAMAPORTNO/api/pull"
echo "$CURLURL"
CURLDATA="{\"name\":\"$MODELNAME\"}"
echo "$CURLDATA"

curl --header "Content-Type: appplication/json" --request POST --data $CURLDATA $CURLURL

echo "Running the vector and model."
#python3 /home/app/ingest.py --MODELNAME $MODELNAME --CHUNKSIZE $CHUNKSIZE --CHUNKOVERLAP $CHUNKOVERLAP --OLLAMAURL $OLLAMAURL --OLLAMAPORTNO $OLLAMAPORTNO --PERSISTDIR $PERSISTDIR --SOURCEDIR $SOURCEDIR
python3 /home/app/ingest.py $MODELNAME $CHUNKSIZE $CHUNKOVERLAP $OLLAMAURL $OLLAMAPORTNO $PERSISTDIR $SOURCEDIR

echo "Running the Private Chat engine."
python3 /home/app/private_chat.py $MODELNAME $CHUNKSIZE $CHUNKOVERLAP $OLLAMAURL $OLLAMAPORTNO $PERSISTDIR $SOURCEDIR