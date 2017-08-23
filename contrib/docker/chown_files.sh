#!/bin/bash

# the docker run commands leave output files with root ownership
# modify the file ownership with the UID of the calling user  

if [ -z $MY_UID ];then
    MY_UID=`id -u`
fi

if [ -z $MY_GID ];then
    MY_GID=`id -g`
fi

if [ -z $MY_ROOT_DIR ];then
    MY_ROOT_DIR=/root/ngraph-test
fi

cd $MY_ROOT_DIR
find . -user root > files_to_chown.txt
cat files_to_chown.txt | xargs chown ${MY_UID} ${1}
cat files_to_chown.txt | xargs chgrp ${MY_GID} ${1}
rm files_to_chown.txt
