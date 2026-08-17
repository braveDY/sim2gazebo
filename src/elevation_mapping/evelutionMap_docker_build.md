# Docker GPU 容器构建与排障记录

本文记录在 Ubuntu 24.04 笔记本上构建 `elevation_mapping_gpu_ros2` Docker 镜像时遇到的问题、根因与解决方法。可作为同类 ROS 2、CUDA、Docker 和本地代理环境的排障参考。

## 最终结果

已成功构建镜像：

```text
elevation_mapping:latest
```

镜像在 `develop` 分支的构建配置下包含：

- ROS 2 Humble
- CUDA 12.1.1 与 cuDNN
- PyTorch CUDA 12.1 轮子
- CuPy 与项目 Python/ROS 依赖
- `elevation_mapping_cupy` 及相关工作区的已编译安装产物

## 环境概览

| 项目 | 最终状态 |
| --- | --- |
| 操作系统 | Ubuntu 24.04.2 LTS |
| GPU | NVIDIA GeForce RTX 3060 Mobile（6 GB 显存） |
| NVIDIA 驱动 | 595.84 |
| Docker | Docker Engine 29.7.2 |
| NVIDIA Container Toolkit | 1.19.1 |
| 项目分支 | `develop` |
| ROS 2/CUDA 镜像配置 | Humble / CUDA 12.1.1 |
| 本地代理 | `http://127.0.0.1:7897` |

> NVIDIA 595 驱动可向后兼容此镜像的 CUDA 12.1，因此主机驱动版本高于容器 CUDA 版本不是问题。

## 构建前准备

### 1. 准备 GPU 驱动

初始检查发现 RTX 3060 使用开源 `nouveau` 驱动，且系统找不到 `nvidia-smi`。这意味着 CUDA、CuPy 及 Docker 的 `--gpus all` 都无法使用。

安装可用驱动后，必须验证：

```bash
nvidia-smi
```

正常输出应包含 NVIDIA GPU 型号、驱动版本和 CUDA 兼容版本。此次环境最终为驱动 `595.84`。

### 2. 确认 Docker daemon 可访问

曾出现以下错误：

```text
permission denied while trying to connect to the docker API at unix:///var/run/docker.sock
```

检查发现 socket 一度是 `nobody:nogroup`，而非正常的 `root:docker`。重启 Docker 服务后恢复：

```bash
sudo systemctl restart docker
ls -l /var/run/docker.sock
docker info
```

预期 socket 属组为 `docker`，且 `docker info` 能同时显示 Client 和 Server 信息。

### 3. 初始化子模块

构建上下文包含平面分割软件包；在项目根目录执行：

```bash
git checkout develop
git submodule update --init --recursive
chmod +x docker/build.sh docker/run.sh
```

## 问题与解决方法

### 问题 1：无法安装 NVIDIA Container Toolkit

Ubuntu 默认 APT 软件源无法找到该软件包：

```text
E: 无法定位软件包 nvidia-container-toolkit
```

原因是该软件包不在 Ubuntu 默认仓库，需要 NVIDIA 官方软件源。添加官方源时还遇到过：

```text
curl: (35) Recv failure: 连接被对方重置
gpg: 找不到有效的 OpenPGP 数据。
```

这说明到 `nvidia.github.io` 的连接被网络或中间代理重置，`gpg` 没有收到密钥数据。切换为可访问该域名的网络或代理后，再添加软件源并安装即可。

安装后检查：

```bash
nvidia-ctk --version
nvidia-container-cli --version
dpkg-query -W -f='${binary:Package}\t${Version}\n' 'nvidia-container-toolkit*'
```

还应配置 Docker runtime：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

验证容器 GPU 访问：

```bash
docker run --rm --gpus all elevation_mapping:latest \
  bash -lc 'nvidia-smi && python3 -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"'
```

预期 CuPy 输出设备数量 `1`。

### 问题 2：BuildKit 无法从 Docker Hub 拉取构建前端

首次执行 `./docker/build.sh` 时，在 Dockerfile 第一行失败：

```text
failed to resolve source metadata for docker.io/docker/dockerfile:1.4
dial tcp [IPv6 地址]:443: i/o timeout
```

Dockerfile 使用了 BuildKit 前端：

```dockerfile
# syntax=docker/dockerfile:1.4
```

根因是本机无法通过 IPv6 稳定访问 Docker Hub。验证结果为：

```bash
curl -4 -I --connect-timeout 15 https://registry-1.docker.io/v2/
curl -6 -I --connect-timeout 15 https://registry-1.docker.io/v2/
```

IPv4 返回 `401` 是正常结果，表示 Registry 可达且等待认证；IPv6 连接失败。

尝试通过 `/etc/gai.conf` 让系统优先 IPv4：

```bash
sudo sed -i 's/^#precedence ::ffff:0:0\/96  100/precedence ::ffff:0:0\/96  100/' /etc/gai.conf
sudo systemctl restart docker
```

但 Docker/BuildKit 仍未稳定采用可用路径，因此最终使用 Docker daemon 的 HTTP 代理。

### 问题 3：Docker daemon 未继承终端代理

终端存在以下代理环境变量：

```text
HTTP_PROXY=http://127.0.0.1:7897
HTTPS_PROXY=http://127.0.0.1:7897
ALL_PROXY=socks5://127.0.0.1:7897
```

终端中的 `curl` 能走代理，但 Docker daemon 是 systemd 服务，不会自动继承这些变量。为 Docker 服务单独配置代理：

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf >/dev/null <<'EOF'
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:7897"
Environment="HTTPS_PROXY=http://127.0.0.1:7897"
Environment="NO_PROXY=localhost,127.0.0.1,::1"
EOF

sudo systemctl daemon-reload
sudo systemctl restart docker
```

确认配置：

```bash
sudo systemctl show --property=Environment docker
docker pull docker/dockerfile:1.4
```

`docker/dockerfile:1.4` 拉取成功后，Docker Hub 的基础镜像拉取恢复正常。

### 问题 4：构建容器无法访问宿主机本地代理

虽然 Docker daemon 已能通过代理拉取镜像，但 Dockerfile 内的 `apt` 和 `pip` 下载仍失败，出现：

```text
ProxyError: Cannot connect to proxy
Failed to establish a new connection: [Errno 111] Connection refused
```

根因是默认构建网络中，容器的 `127.0.0.1` 指向容器自身，而代理服务监听在宿主机的 `127.0.0.1:7897`。

解决方法是在构建时使用宿主机网络，并将代理作为构建参数传入：

```bash
DOCKER_BUILDKIT=1 docker build \
  --network host \
  --file docker/Dockerfile.x64 \
  --target runtime \
  --tag elevation_mapping:latest \
  --build-arg ROS_DISTRO=humble \
  --build-arg USERNAME=ros \
  --build-arg USER_UID="$(id -u)" \
  --build-arg USER_GID="$(id -g)" \
  --build-arg INSTALL_EMCUPY_ROSDEPS=true \
  --build-arg HTTP_PROXY="$HTTP_PROXY" \
  --build-arg HTTPS_PROXY="$HTTPS_PROXY" \
  --build-arg NO_PROXY="localhost,127.0.0.1,::1,archive.ubuntu.com,security.ubuntu.com" \
  --build-arg http_proxy="$http_proxy" \
  --build-arg https_proxy="$https_proxy" \
  --build-arg no_proxy="localhost,127.0.0.1,::1,archive.ubuntu.com,security.ubuntu.com" \
  .
```

`--network host` 仅用于构建网络。它使构建步骤中的 `127.0.0.1:7897` 指向宿主机代理。

### 问题 5：代理对 Ubuntu APT 源间歇性返回 502

构建容器成功使用代理后，APT 又出现过以下错误：

```text
502 Bad Gateway [IP: 127.0.0.1 7897]
E: The repository 'http://archive.ubuntu.com/ubuntu jammy InRelease' is not signed.
```

也曾在 `security.ubuntu.com` 上出现相同的 502。后一个“未签名”错误不是签名本身有问题，而是 APT 未成功下载 `InRelease` 文件后的连带报错。

解决方式是在 `NO_PROXY` 与 `no_proxy` 中排除以下域名，使它们直连：

```text
archive.ubuntu.com,security.ubuntu.com
```

其余需要跨境访问且更适合使用代理的资源（Docker Hub、GitHub、PyPI、PyTorch）仍保持走代理。

### 问题 6：PyTorch 下载缓慢并触发哈希校验失败

未稳定使用代理时，PyTorch 下载速度仅约 `10.5 kB/s`，中途报错：

```text
ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE
```

不要通过关闭 pip 哈希校验来规避此问题。该错误表示下载内容不完整或异常，应先修复网络路径。采用宿主机网络和构建代理参数后，PyTorch 层成功完成并被 Docker 缓存。

### 问题 7：`flake8-blind-except` 与旧版 `packaging` 不兼容

网络问题解决后，构建在 Python 工具依赖层失败：

```text
TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'
```

根因是 Ubuntu 22.04 系统内的 `packaging` 版本过旧，而 `flake8-blind-except` 的元数据生成使用了较新 API。

在 [docker/Dockerfile.x64](../docker/Dockerfile.x64) 的 PyTorch 安装步骤之后加入：

```dockerfile
RUN python3 -m pip install -U 'packaging>=22'
```

此改动让 PyTorch 层保持可复用缓存，只重新执行后续 Python 依赖层。最终所有 27 个构建步骤均成功完成。

## 缓存与重试建议

- 构建中途失败时，优先修复根因后使用相同命令重试。
- 不要轻易添加 `--no-cache`；CUDA 镜像、APT 依赖和 PyTorch 都很大，BuildKit 可复用已成功的层。
- 更改 `--network` 或 Dockerfile 中某一层会影响相关层的缓存，但不会必然重下载所有基础镜像。
- 遇到 APT 的单次 502 时可重试；若持续发生，按本文件将不稳定的 APT 域名加入 `NO_PROXY`。
- 遇到下载哈希不一致时，应检查代理或网络稳定性，不应禁用完整性校验。

## 构建后运行

确认 GPU 验证通过后，在项目根目录运行：

```bash
./docker/run.sh
```

脚本会挂载本机的高程地图与平面分割配置目录，并通过 `--gpus all` 启动交互式容器。

容器内启动节点：

```bash
ros2 launch elevation_mapping_cupy elevation_mapping.launch.py use_sim_time:=False
```

## 注意事项

- 本次构建使用的是 `develop` 分支。该分支的 `docker/build.sh` 固定构建 ROS 2 Humble 与 CUDA 12.1.1；不要将其与其他分支的 Jazzy/CUDA 12.8 配置混淆。
- 本机代理地址为回环地址，未包含账号密码。若代理 URL 含凭据，不应将其提交到仓库、Dockerfile 或公开文档。
- Docker daemon 的 systemd 代理配置会影响该主机上的 Docker 网络访问；更换代理端口后，需同步修改 `/etc/systemd/system/docker.service.d/http-proxy.conf` 并重启 Docker。
