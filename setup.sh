#!/bin/bash

pip3 install -r requirements.txt
git clone https://github.com/ashleykleynhans/civitai-downloader.git && cd civitai-downloader && \
chmod +x download.sh && bash download.sh https://civitai.com/api/download/models/60568 ../ && cd .. && \
rm -rf civitai-downloader
pip install diffusers transformers --upgrade