# 稳定绿色包说明

本项目的稳定绿色包目标是：

- 解压即用
- 可以直接复制到其他 Windows x64 + NVIDIA 电脑运行
- 不依赖目标机系统 Python
- 不依赖目标机系统 ffmpeg

它不是单文件安装器，也不是单文件 `exe`，而是参考 ComfyUI portable 的整目录分发模式。

## 目标目录结构

```text
LiveTalking-portable-win64-nvidia-cu128/
  app.py
  start.ps1
  start.bat
  start_nvidia_gpu.bat
  configs/
  data/
  docs/
  livetalking/
  models/
  python/
  runtime/
  scripts/
  tools/
    ffmpeg/
      bin/
        ffmpeg.exe
        ffprobe.exe
  web/
  wav2lip/
```

## 目录职责

- `python/`
  - 便携 Python 3.10 运行时
  - 已包含项目依赖
- `tools/ffmpeg/bin/`
  - 便携 ffmpeg / ffprobe
- `models/`
  - ASR、TTS、Wav2Lip 模型与权重
- `data/avatars/`
  - 全部当前头像资源
- `runtime/logs/`
  - 运行日志目录
- `runtime/tmp/`
  - 临时文件目录

## 开发机构建步骤

### 1. 准备本地资源

确认以下内容已经位于项目目录内：

- `models/`
- `data/avatars/`

### 2. 准备便携运行时

执行：

```powershell
.\scripts\prepare_portable_runtime.ps1
```

默认行为：

- 自动优先使用 `py -3.10`
- 自动查找本机 `ffmpeg`
- 复制 Python 3.10 到项目根目录 `python/`
- 复制 ffmpeg 到项目根目录 `tools/ffmpeg/bin/`
- 把 `requirements-torch-cu128.txt` 和 `requirements.txt` 安装到 `python/`

如果自动发现失败，可以显式指定：

```powershell
.\scripts\prepare_portable_runtime.ps1 `
  -PythonExe "C:\Users\admin\AppData\Local\Programs\Python\Python310\python.exe" `
  -FfmpegBinDir "D:\Program Files\ffmpeg\bin"
```

### 3. 生成稳定绿色包

执行：

```powershell
.\scripts\build_portable_package.ps1
```

输出：

- `dist/LiveTalking-portable-win64-nvidia-cu128/`
- `dist/LiveTalking-portable-win64-nvidia-cu128.zip`（仅在显式开启压缩时生成）

## 目标机使用方式

1. 解压 `LiveTalking-portable-win64-nvidia-cu128.zip`
2. 双击 `start_nvidia_gpu.bat`
3. 浏览器打开：

`http://127.0.0.1:8010/`

## 打包前校验项

打包脚本会校验：

- `python/python.exe`
- `python/python310.dll`
- `python/Lib/site-packages/`
- `tools/ffmpeg/bin/ffmpeg.exe`
- `tools/ffmpeg/bin/ffprobe.exe`
- `models/` 中核心模型文件
- `configs/app.yaml` 中默认 `avatar_id` 对应的头像目录

## 运行约束

- 首版只支持 Windows x64
- 首版只支持 NVIDIA GPU
- 首版不提供 CPU 包
- 首版不提供更新脚本
- 首版不负责离线封装外部 LLM 服务

LLM 相关 `base_url`、`api_key`、`model` 仍按现有配置方式由用户自行设置。
