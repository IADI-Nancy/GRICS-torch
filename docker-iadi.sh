IMAGE_NAME=virtus.iadi.lan:8123/iadi/grics-torch
IMAGE_VERSION=2.3.0-cuda12.1-cudnn8-devel

IMAGE="${IMAGE_NAME}:${IMAGE_VERSION}"
CONTAINER_SUFFIX=grics-torch

WKDIR=$(pwd)

# !!!!!!!!!!!!!!!!!!
EXTRAS=( -v ~/.cicit-nancy:/home/pyuser/.cicit-nancy -v .bashrcoverride:/home/pyuser/.bashrcoverride )
# put this in extras
# if you want to add something into .bashrc without loosing it
# -v .bashrcoverride:/home/pyuser/.bashrcoverride
# if you use ArchiMedConnector
# -v ~/.cicit-nancy:/home/pyuser/.cicit-nancy

GPU_NUMBER=0
#########################################

# ????????????
#avoid root ownership on these files
mkdir -p $WKDIR
mkdir -p ~/.ssh
mkdir -p ~/.cicit-nancy
mkdir -p ~/.codex
touch ~/.gitconfig

MOUNTS=( -v $WKDIR:/home/pyuser/wkdir -v ~/.ssh:/home/pyuser/.ssh -v ~/.gitconfig:/home/pyuser/.gitconfig -v ~/.codex:/home/pyuser/.codex )
GPU=( --gpus device=$GPU_NUMBER )
#GPU=( )

# https://man7.org/linux/man-pages/man7/shm_overview.7.html
SHM=( )
#SHM=( --shm-size=256mb )

USERCUT=$(echo $USER | cut -d@ -f1)
DOCKER_NAME="${USERCUT}-${CONTAINER_SUFFIX}"

HTTP_PROXY="http://proxy-pc.chu-nancy.fr:8080/"
HTTPS_PROXY="http://proxy-pc.chu-nancy.fr:8080/"
NO_PROXY=".iadi.lan,localhost,127.0.0.1,192.168.139.0/24"

case "$1" in
run)	
	echo "Run docker " $DOCKER_NAME
	if [ -z "$(docker ps -a | grep $DOCKER_NAME)" ]; then
		docker run -d --network host ${EXTRAS[@]} ${MOUNTS[@]} ${GPU[@]} ${SHM[@]} --name ${DOCKER_NAME} \
			-e USERCUT=${USERCUT} -e UID=$(id -u) -e GID=$(id -g) \
			-e HTTP_PROXY="${HTTP_PROXY}" -e HTTPS_PROXY="${HTTPS_PROXY}" \
			-e NO_PROXY="${NO_PROXY}" -e http_proxy="${HTTP_PROXY}" \
			-e https_proxy="${HTTPS_PROXY}" -e no_proxy="${NO_PROXY}" \
			$IMAGE 
	else
		docker start ${DOCKER_NAME}
	fi
	printf "Waiting for startup..."
	while [ -z "$(docker exec ${DOCKER_NAME} sh -c 'ls -a /home/pyuser | grep -F .bashrc')" ]; do
		printf "."
		sleep 1
	done
	echo 'OK'
	echo "To use your container in VSCode, please copy/paste following json in the image-config-file of ${IMAGE} :"
	echo '{
	"workspaceFolder": "/home/pyuser",
	"updateRemoteUserUID": false,
	"containerUser": "root",
	"remoteUser": "pyuser",
	"extensions": [
		"ms-python.python",
		"ms-python.vscode-pylance",
		"ms-toolsai.jupyter",
		"ms-toolsai.jupyter-keymap",
		"ms-toolsai.jupyter-renderers"
	]
}'
	;;

exec)
	PARAMS=${@:2}
	if [ -z ${2} ]; then
		PARAMS="/bin/bash"
	fi
	docker exec --user=$(id -u) $DOCKER_NAME ${PARAMS}
	;;

exec-it)
	PARAMS=${@:2}
	if [ -z ${2} ]; then
		PARAMS="/bin/bash"
	fi
	docker exec --user=$(id -u) -it $DOCKER_NAME ${PARAMS}
	;;

exec-it-root)
	PARAMS=${@:2}
	if [ -z ${2} ]; then
		PARAMS="/bin/bash"
	fi
	docker exec --user=0 -it $DOCKER_NAME ${PARAMS}
	;;

start)	
	docker start $DOCKER_NAME
	;;

stop)	
	docker stop $DOCKER_NAME
	;;

rm)	
	docker stop $DOCKER_NAME
	docker rm $DOCKER_NAME
	;;

build)
	docker build -f build/Dockerfile-iadi -t $IMAGE --build-arg PYTORCH_VER=$IMAGE_VERSION --network=host --rm=true build/
	;;

build-nocache)
	docker build -f build/Dockerfile-iadi -t $IMAGE --build-arg PYTORCH_VER=$IMAGE_VERSION --network=host --no-cache --rm=true build/
	;;

push)
	docker push -a $IMAGE_NAME
	;;

*)	
	echo "Usage : start-docker.sh (run|exec [command]|exec-it [command]|exec-it-root [command]|stop|rm|prune|build|build-nocache)"
	;;
esac














#docker run --detach --name $DOCKER_NAME -e PYUSERID=$(id -u $USER) -e PYGROUPID=$(id -g $USER) --volume=$(pwd)/data:/mnt/data --volume=$(pwd)/code:/mnt/code  --volume=$(pwd)/records:/mnt/records $IMAGE_NAME


#docker run --detach --name $DOCKER_NAME $IMAGE_NAME#--volume=$(pwd)/data:/mnt/data --volume=$(pwd)/code:/mnt/code  --volume=$(pwd)/records:/mnt/records $IMAGE_NAME
