# LiveTalking Local

本项目基于 [lipku/LiveTalking](https://github.com/lipku/LiveTalking) 整理，当前保留并维护的主链路为：

`wav2lip + Paraformer + sherpa-onnx / EdgeTTS + LLM + WebRTC`

当前仓库同时支持两种使用方式：

- 开发运行：使用项目根目录 `.venv`
- 稳定绿色包：使用项目根目录 `python/`、`tools/ffmpeg/`，打成可复制的便携目录包

## 项目结构

```text
app.py
configs/
data/
docs/
livetalking/
models/
runtime/
scripts/
tools/
wav2lip/
web/
```

结构说明见 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)。

## 运行要求

- Windows x64
- Python 3.10
- NVIDIA GPU
- 浏览器
- 所有模型、头像、运行时文件均位于项目目录内

## 开发运行

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
.\.venv\Scripts\python.exe -m pip install "setuptools<81"
.\.venv\Scripts\python.exe -m pip install -r .\requirements-torch-cu128.txt
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe .\app.py
```

启动后访问：

`http://127.0.0.1:8010/`

## 稳定绿色包

稳定绿色包不是单文件 `exe`，而是参考 ComfyUI portable 的整目录分发模式：

- 项目内 `python/` 提供独立 Python 运行时
- 项目内 `tools/ffmpeg/bin/` 提供独立 ffmpeg
- 项目内 `models/`、`data/avatars/` 提供运行资源
- 启动入口为 `start_nvidia_gpu.bat`

构建步骤：

1. 准备便携运行时：

```powershell
.\scripts\prepare_portable_runtime.ps1
```

2. 生成稳定绿色包：

```powershell
.\scripts\build_portable_package.ps1
```

产物：

- `dist/LiveTalking-portable-win64-nvidia-cu128/`
- `dist/LiveTalking-portable-win64-nvidia-cu128.zip`

详细说明见 [docs/PORTABLE_PACKAGE.md](docs/PORTABLE_PACKAGE.md)。

## 配置说明

- 默认运行配置：`configs/app.yaml`
- 配置参考模板：`configs/app.example.yaml`
- 建议通过环境变量注入密钥：
  - `LIVETALKING_OPENAI_API_KEY`
  - `OPENAI_API_KEY`

`configs/app.yaml` 继续使用项目内相对路径，不依赖开发机绝对路径。

## 资源目录约定

- 头像素材：`data/avatars/<avatar_id>/`
- ASR 模型：`models/asr/...`
- TTS 模型：`models/tts/...`
- Wav2Lip 权重：`models/avatar/wav2lip256/wav2lip256.pth`
- 运行日志：`runtime/logs/`
- 临时文件：`runtime/tmp/`

## 辅助脚本

- 资源准备：`scripts/prepare_local_assets.ps1`
- 便携运行时准备：`scripts/prepare_portable_runtime.ps1`
- 稳定绿色包打包：`scripts/build_portable_package.ps1`
