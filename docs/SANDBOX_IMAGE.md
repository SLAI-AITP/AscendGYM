# 官方镜像使用

官方 AscendForge 镜像托管在独立镜像平台，不随开源代码仓库分发。使用前需申请相应访问权限，再从平台拉取镜像。无需官方镜像的接入方式见[自建环境](SELF_HOSTED_ENVIRONMENT.md)。

## 1. 获取镜像

镜像平台地址：`<IMAGE_PLATFORM_URL>`（待填写）。访问该平台并按平台说明完成权限申请与授权。

获得权限后，将下方 `SANDBOX_IMAGE` 设置为平台提供的完整镜像引用，包含仓库域名、命名空间、镜像名称与版本标签，或指定 digest。平台网页 URL 与 Docker 镜像引用用途不同，不要将带 `https://` 的网页地址直接用于 `docker pull`。

```bash
export SANDBOX_IMAGE='<IMAGE_REFERENCE>'
```

如平台要求 Docker 登录，先按平台提供的认证方式登录镜像仓库。对于上述包含仓库域名的完整镜像引用，可使用：

```bash
docker login "${SANDBOX_IMAGE%%/*}"
```

然后拉取镜像：

```bash
docker pull "$SANDBOX_IMAGE"
```

下文启动命令复用 `SANDBOX_IMAGE`，无需重复填写镜像名称。评测记录中应保存实际使用的镜像引用和版本；镜像平台地址用于获取访问说明，不是 bridge 的编译服务地址。

## 2. 启动服务

以下启动示例沿用 Linux aarch64、Ascend 910C 参考环境。实际镜像支持范围以平台上的版本说明为准。宿主机需有兼容的驱动、可用的 NPU 和容器设备访问能力；镜像不能代替宿主机驱动。仅运行 bridge 的客户端不需要这些设备。

以下为单 worker 启动示例。设备名和驱动目录必须按宿主机实际配置调整，仅映射分配给服务的设备；若平台使用专用容器运行时，应使用平台提供的设备映射方式。

```bash
docker run -d --name ascendgym-sandbox \
  -p 9000:8080 \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  "$SANDBOX_IMAGE" \
  uvicorn sandbox.server.server:app --host 0.0.0.0 --port 8080
```

当前参考镜像默认命令为交互式 shell，以下示例显式传入服务启动命令；平台后续版本若调整入口，应按相应版本说明启动。示例将宿主机 `9000` 映射到容器 `8080`；请把服务放在受控网络中，端口开放范围由部署方配置。

```bash
docker logs --tail 100 ascendgym-sandbox
curl --fail http://127.0.0.1:9000/ping
```

启动命令是部署示例，不能代替目标宿主机的 NPU 验收。已有参考镜像的容器 API 检查和节点运行环境测试记录，不等于所拉取版本在目标部署上的 NPU 验收结果。请在目标部署上执行下一节检查。

## 3. 对接 bridge

在客户端把 `<sandbox-host>` 替换为客户端可访问的宿主机地址。客户端在另一台机器上时，不要使用 `127.0.0.1` 指向远端服务。

```bash
export ASCEND_GYM_WORKSPACE=/path/to/AscendGYM
export ASCEND_GYM_DATA_ROOT="$ASCEND_GYM_WORKSPACE/datasets"
export SANDBOX_URL="http://<sandbox-host>:9000"
cd "$ASCEND_GYM_WORKSPACE/bridge"
python3 op_manager.py smoke --op add_rms_norm --cases all --profiling-runs 5
```

核对全部测试轮次、错误日志、非空 profiling 和耗时解析结果后，再按[根 README](../README.md#quick-start)配置模型并运行评测。单 worker 不提供多节点调度；需要指定节点或 NPU 的调度约束时，应接入支持这些约束的调度服务。

源码、测试和 golden 由 bridge 上传，不需要将整个开源仓库或客户端绝对路径挂载进服务容器。任务产物保存在客户端的评测输出目录；容器重建前如需保留服务日志，应另行收集。
