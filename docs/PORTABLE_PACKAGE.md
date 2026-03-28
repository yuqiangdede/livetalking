# 绿色包说明

本项目的绿色包目标是：

- 解压即用
- 可直接复制到另一台 Windows x64 + NVIDIA 机器运行
- 不依赖目标机系统 Python
- 不依赖目标机系统 ffmpeg
- 所有模型、资源、依赖都位于项目目录内

它不是单文件安装器，也不是单文件 `exe`，而是整目录分发模式。

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
  - 项目内独立 Python 3.10 运行时
- `tools/ffmpeg/bin/`
  - 项目内独立 ffmpeg / ffprobe
- `models/`
  - ASR、TTS、Wav2Lip 模型与权重
- `data/avatars/`
  - 全部头像素材
- `runtime/logs/`
  - 运行日志目录
- `runtime/tmp/`
  - 临时文件目录

## 构建步骤

### 1. 准备项目内模型与资源

打包前确认以下内容已位于项目目录内：

- `models/asr/...`
- `models/avatar/...`
- `models/tts/sherpa-onnx-vits-zh-ll`
- `models/tts/vits-melo-tts-zh_en`
- `data/avatars/...`

如需启用 Qwen3-TTS，还需要：

- `models/tts/qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `models/tts/qwen/Qwen3-TTS-Tokenizer-12Hz`

### 2. 准备项目内便携运行时

```powershell
.\scripts\prepare_portable_runtime.ps1
```

默认行为：

- 自动定位 Python 3.10
- 自动定位 ffmpeg
- 复制 Python 到项目根目录 `python/`
- 复制 ffmpeg 到项目根目录 `tools/ffmpeg/bin/`
- 安装 `requirements-torch-cu128.txt` 与 `requirements.txt`
- 默认不安装 Qwen / Coqui 的可选依赖
- 如需启用 Qwen3-TTS，显式传入 `-IncludeQwenTts`
- 如需启用 Coqui XTTS v2，显式传入 `-IncludeCoquiTts`
- 打包产物会保留上述 TTS 依赖清单，方便目标机重新执行准备脚本

### 3. 生成绿色包目录

```powershell
.\scripts\build_portable_package.ps1
```

默认仅生成目录：

- `dist/LiveTalking-portable-win64-nvidia-cu128/`
- 默认不把 Qwen / Coqui 模型与参考音频打进包体
- 如需把 Qwen3-TTS 带入绿色包，显式传入 `-IncludeQwenTts`
- 如需把 Coqui XTTS v2 带入绿色包，显式传入 `-IncludeCoquiTts`

如需同时生成 zip：

```powershell
.\scripts\build_portable_package.ps1 -SkipZip:$false
```

## Qwen3-TTS 的打包规则

- Qwen 模型是可选资源，不是必选资源
- 默认绿色包不包含 Qwen 模型和 tokenizer
- 只有显式传入 `-IncludeQwenTts`，打包脚本才会尝试把模型和 tokenizer 一起带入绿色包
- 如果绿色包里没带入 Qwen，运行时会自动回落到 `vits_zh`
- 如果默认配置 `providers.tts=qwen3_customvoice`，但没有显式开启对应开关，打包会直接失败
- 如果显式开启了对应开关，但项目内缺少 Qwen 模型或 tokenizer，打包也会失败

## Coqui XTTS v2 的打包规则

- Coqui XTTS v2 是可选资源，不是必选资源
- 默认绿色包不包含 Coqui XTTS v2 模型和参考音频
- 只有显式传入 `-IncludeCoquiTts`，打包脚本才会尝试把模型和参考音频带入绿色包
- 如果绿色包里没带入 Coqui，运行时会自动回落到 `vits_zh`
- 如果默认配置 `providers.tts=coqui_xtts_v2`，但没有显式开启对应开关，打包会直接失败
- 如果显式开启了对应开关，但项目内缺少 Coqui 模型或参考音频，打包也会失败

## 目标机使用方式

1. 解压绿色包目录或 zip
2. 双击 `start_nvidia_gpu.bat`
3. 浏览器打开 `http://127.0.0.1:8010/`

## 打包校验项

打包脚本会校验：

- `python/python.exe`
- `python/python310.dll`
- `python/Lib/site-packages/`
- `tools/ffmpeg/bin/ffmpeg.exe`
- `tools/ffmpeg/bin/ffprobe.exe`
- `models/` 中核心 ASR / sherpa TTS / Wav2Lip 文件
- `configs/app.yaml` 中默认 `avatar_id` 对应的头像目录
- 若 Qwen 模型目录存在，则额外校验：
  - `models/tts/qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/config.json`
  - `models/tts/qwen/Qwen3-TTS-Tokenizer-12Hz/config.json`

## 运行约束

- 当前只支持 Windows x64
- 当前只支持 NVIDIA GPU
- 当前不提供 CPU 绿色包
- 当前不在运行时自动联网下载 Qwen 模型
- 若要使用 Qwen3-TTS，必须先通过项目内脚本下载到项目目录内
