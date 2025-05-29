#!/bin/bash

docker login -u \$oauthtoken nvcr.io

docker buildx build --platform linux/amd64 -t srijitnair254/bionemo_dbx_amd64:0.1 -f dockerfile .

docker login -u srijitnair254 docker.io 

docker push srijitnair254/bionemo_dbx_amd64:0.1
