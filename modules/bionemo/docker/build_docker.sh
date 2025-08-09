#!/bin/bash

docker login -u \$oauthtoken nvcr.io

docker buildx build --platform linux/amd64 -t <your tag> -f dockerfile .

docker login <to your container repo> 

docker push <your tag>
