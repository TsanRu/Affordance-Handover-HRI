#!/bin/bash
# Run the ur_python_utilities ROS 2 Docker container and open a shell.
#
# Usage: ./docker/RUN-DOCKER.sh [optional: project_name]
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export DOCKER_RUNTIME=${DOCKER_RUNTIME:-nvidia}

PROJECT=$1
if [ -z "${PROJECT}" ]; then
  PROJECT=${USER}
fi
CONTAINER="${PROJECT}-ur-python-utilities-1"
echo "$0: PROJECT=${PROJECT}"
echo "$0: CONTAINER=${CONTAINER}"

docker compose -p ${PROJECT} -f ./docker/docker-compose.yml up -d

xhost +

docker exec -it ${CONTAINER} bash
