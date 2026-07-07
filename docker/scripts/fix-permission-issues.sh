#!/bin/bash

chown -R ${HOST_UID}:${HOST_GID} /root/.ros/ 2>/dev/null || true
chown -R ${HOST_UID}:${HOST_GID} /root/ws/
