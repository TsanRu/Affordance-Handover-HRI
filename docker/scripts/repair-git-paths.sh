#!/bin/bash

cd /root/ws/
find . -type f -iname '.git' | \
  while read -r f ; do
    if grep -qE '^gitdir: /' "$f" ; then
      echo "Fix a full path in $f."
      RELATIVE=$(echo "${f%/*/.git}" | sed -ne 's/\([^/]*\)/../gp')
      sed -i -e 's@ \([^ ]*\)/\.git@ '"${RELATIVE}"'/.git@' "$f"
    fi
  done
