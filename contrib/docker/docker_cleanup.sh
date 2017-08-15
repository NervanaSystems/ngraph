#!/bin/bash

# list active docker containers
echo "Active docker containers..."
docker ps -a
echo

# clean up old docker containers
echo "Removing Exited docker containers..." 
docker ps -a | grep Exited | cut -f 1 -d ' ' | xargs docker rm -f ${1}
echo

#list docker images for ngraph
echo "Docker images for ngraph..."
docker images ngraph_* 
echo

# clean up docker images no longer in use
echo "Removing docker images for ngraph..." 
docker images -qa ngraph_* | xargs docker rmi -f ${1}
