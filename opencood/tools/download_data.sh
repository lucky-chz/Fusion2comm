#!/bin/bash
  
# cd scratch place

# Download zip dataset from Google Drive
filename='validata.zip'
fileid='1M4pG-fdPs-EWMLZpc1yl-bqUcJ6yg4zz'
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm ./cookie
  
# Unzip
unzip -q ${filename}
rm ${filename}
  
# cd out
cd