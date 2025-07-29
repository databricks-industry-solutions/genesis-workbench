#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <path_to_working_directory> <env>"
    echo 'Example: destroy core dev '
    exit 1
fi

CWD=$1
ENV=$2

cd modules/$CWD
./destroy.sh $ENV

