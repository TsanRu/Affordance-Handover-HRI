#!/bin/bash
# Build the ur_python_utilities ROS 2 Jazzy Docker image.
#
# Usage: ./docker/BUILD-DOCKER-IMAGE.sh [optional: project_name]
#   project_name defaults to $USER (docker compose project prefix).
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DOCKER_PROJECT=$1
if [ -z "${DOCKER_PROJECT}" ]; then
  DOCKER_PROJECT=${USER}
fi
DOCKER_CONTAINER="${DOCKER_PROJECT}-ur-python-utilities-1"
echo "$0: DOCKER_PROJECT=${DOCKER_PROJECT}"
echo "$0: DOCKER_CONTAINER=${DOCKER_CONTAINER}"

EXISTING_DOCKER_CONTAINER_ID=$(docker ps -aq -f name=${DOCKER_CONTAINER})
if [ ! -z "${EXISTING_DOCKER_CONTAINER_ID}" ]; then
  echo "Stop the container ${DOCKER_CONTAINER} with ID: ${EXISTING_DOCKER_CONTAINER_ID}."
  docker stop ${EXISTING_DOCKER_CONTAINER_ID}
  echo "Remove the container ${DOCKER_CONTAINER} with ID: ${EXISTING_DOCKER_CONTAINER_ID}."
  docker rm ${EXISTING_DOCKER_CONTAINER_ID}
fi

export DOCKERFILE_COMMIT_SHORT_SHA="$(git log -n 1 --pretty=format:%h docker/Dockerfile 2>/dev/null || echo unknown)"

docker compose -p ${DOCKER_PROJECT} -f ./docker/docker-compose.yml build
