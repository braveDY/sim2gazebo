#!/usr/bin/env bash

EM_CONTAINER="${EM_CONTAINER:-em_gpu_humble}"
EM_IMAGE="${EM_IMAGE:-elevation_mapping:latest}"
EM_HOST_DIR="${EM_HOST_DIR:-/home/brave/sim2sim}"
EM_MOUNT_DIR="${EM_MOUNT_DIR:-/sim2sim}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${EM_HOST_DIR}/rl_sar/src/elevation_mapping}"

em_exists() { docker ps -a --format '{{.Names}}' | grep -Fxq "${EM_CONTAINER}"; }
em_running() { docker ps --format '{{.Names}}' | grep -Fxq "${EM_CONTAINER}"; }

em_allow_x11() {
    export DISPLAY="${DISPLAY:-:0}"
    export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"
    command -v xhost >/dev/null 2>&1 && \
        xhost +SI:localuser:"$(id -un)" >/dev/null 2>&1 || true
}

em_create() {
    em_exists && { echo "容器已存在：${EM_CONTAINER}"; return 0; }
    [[ -d "${EM_HOST_DIR}" ]] || { echo "挂载目录不存在：${EM_HOST_DIR}" >&2; return 1; }
    docker image inspect "${EM_IMAGE}" >/dev/null 2>&1 || { echo "镜像不存在：${EM_IMAGE}" >&2; return 1; }
    em_allow_x11

    docker run -d \
        --name "${EM_CONTAINER}" --restart unless-stopped \
        --privileged --network host --ipc host --gpus all \
        --workdir "${EM_MOUNT_DIR}" \
        -e "DISPLAY=${DISPLAY}" \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e QT_X11_NO_MITSHM=1 -e ROS_DOMAIN_ID=1 \
        -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
        -v "${EM_HOST_DIR}:${EM_MOUNT_DIR}:rw" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        "${EM_IMAGE}" sleep infinity
}

em_start() {
    if ! em_exists; then
        em_create || return 1
    fi
    em_allow_x11
    em_running || docker start "${EM_CONTAINER}" >/dev/null
}

em_stop() { docker stop "${EM_CONTAINER}"; }
em_enter() {
    em_start || return 1
    docker exec -it --workdir "${EM_MOUNT_DIR}" \
        -e "DISPLAY=${DISPLAY:-:0}" -e "QT_X11_NO_MITSHM=${QT_X11_NO_MITSHM:-1}" \
        "${EM_CONTAINER}" bash
}
em_logs() { docker logs --tail 100 -f "${EM_CONTAINER}"; }
em_status() { docker ps -a --filter "name=^/${EM_CONTAINER}$"; }
em_remove() { docker rm -f "${EM_CONTAINER}"; }

em_help() {
    echo "命令：em_create em_start em_stop em_enter em_logs em_status em_remove"
}
