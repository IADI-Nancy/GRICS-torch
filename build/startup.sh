#!/bin/bash

: ${UID:=0}
: ${GID:=0}
: ${USERCUT:=}

if [ ! -f "/home/pyuser/.bashrc" ] && [ "$UID" -ne "0" ] ; then
	addgroup --gid ${GID} pyuser
	adduser --uid ${UID} --gid ${GID} --no-create-home --gecos "" --disabled-password pyuser
	usermod -aG sudo pyuser
	cp -r /etc/skel/. /home/pyuser

	chown -R pyuser:pyuser /home/pyuser

	echo '
if [ -f ~/.bashrcoverride ]; then
  source ~/.bashrcoverride
fi
' >> /home/pyuser/.bashrc
fi

echo "End of starting"
tail -f /dev/null
