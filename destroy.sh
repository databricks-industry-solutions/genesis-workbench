#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <path_to_working_directory> <env>  <additional build variables>"
    echo 'Example: destroy core dev --var="dev_user_prefix=scn"'
    exit 1
fi

CWD=$1
ENV=$2
EXTRA_PARAMS=${@: 3}

cd modules/$CWD
./destroy.sh $ENV $EXTRA_PARAMS

