#!/bin/bash

################################################################################

# Start the D-Bus daemon.
service dbus start

################################################################################

# Support Intel 3D acceleration (when no Nvidia GPU drivers are found).
if [ "$DOCKER_RUNTIME" = "runc" ]; then
  rm -f /etc/ld.so.conf.d/nvidia.conf /etc/ld.so.conf.d/glvnd.conf
  ldconfig
fi

################################################################################

tail -f /dev/null
