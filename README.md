# LiveTalking Local

本项目基于 [lipku/LiveTalking](https://github.com/lipku/LiveTalking) 整理，当前保留并维护的主链路为：

`wav2lip + Paraformer + sherpa-onnx / Qwen3-TTS / Coqui XTTS v2 / EdgeTTS / pyttsx3 + LLM + WebRTC`

当前仓库支持两种使用方式：

- 开发运行：使用项目根目录下的 `.venv`
- 绿色包运行：使用项目根目录下的 `python/` 和 `tools/ffmpeg/`

所有模型、权重、资源、缓存、上传文件和运行时目录都必须放在项目目录内，不依赖项目目录外资源。

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
- 浏览器支持 WebRTC
- 如使用 Qwen3-TTS 或 Coqui XTTS v2，推荐 NVIDIA CUDA GPU
- Python 虚拟环境固定放在项目根目录 `.venv`

## 开发运行

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
.\.venv\Scripts\python.exe -m pip install "setuptools<81"
.\.venv\Scripts\python.exe -m pip install -r .\requirements-torch-cu128.txt
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe -m pip install -r .\requirements-qwen-tts.txt
.\.venv\Scripts\python.exe -m pip install -r .\requirements-coqui-tts.txt
.\.venv\Scripts\python.exe -m pip install "coqui-tts[codec]"
.\.venv\Scripts\python.exe .\app.py
```

启动后访问：

`http://127.0.0.1:8010/`

## 已支持的 TTS

- `pyttsx3`
  - 本地系统 TTS
  - 无显卡要求
  - 当前界面固定提供 3 个选项：系统默认音色、Huihui（中文）、Zira（英文）
- `edgetts`
  - 在线多语言 TTS
  - 无显卡要求
- `vits_zh`
  - 本地 sherpa-onnx 中文 TTS
  - 使用 `models/tts/sherpa-onnx-vits-zh-ll`
  - CPU 可运行
- `vits_melo_zh_en`
  - 本地 sherpa-onnx 中英双语 TTS
  - 使用 `models/tts/vits-melo-tts-zh_en`
  - CPU 可运行
- `qwen3_customvoice`
  - 本地 Qwen3-TTS 0.6B CustomVoice
  - 支持预置音色和风格词
  - 推荐 NVIDIA CUDA GPU，CPU 可运行但较慢
- `coqui_xtts_v2`
  - 本地 Coqui XTTS v2 多语言语音克隆
  - 使用项目内模型目录和项目内参考音频
  - 当前界面仅开放中文、英文
  - 推荐 NVIDIA CUDA GPU，CPU 可运行但较慢

## 所有 TTS 的播放方式

当前所有 TTS 已统一改为：

- 按中英文标点分段合成
- 第一段先出声，不等待整句全部生成完成
- 首段音频先入队，后续分段继续补充

这套行为已覆盖：

- `pyttsx3`
- `edgetts`
- `vits_zh`
- `vits_melo_zh_en`
- `qwen3_customvoice`
- `coqui_xtts_v2`

## Qwen3-TTS 预置音色

- `Vivian`
- `Serena`
- `Uncle_Fu`
- `Dylan`
- `Eric`
- `Ryan`
- `Aiden`
- `Ono_Anna`
- `Sohee`

## Coqui XTTS v2 准备方式

### 1. 安装依赖

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-coqui-tts.txt
.\.venv\Scripts\python.exe -m pip install "coqui-tts[codec]"
```

### 2. 下载模型到项目目录

```powershell
.\scripts\download_coqui_xtts_v2_models.ps1
```

默认会下载到：

- `models/tts/coqui/xtts_v2/`

目录内至少应包含：

- `config.json`
- `*.pth` 或 `*.pt` 模型文件

### 3. 准备参考音频

默认参考音频路径：

- `data/tts/coqui_xtts_v2/reference.wav`

也可以在页面设置中上传参考 wav，文件会保存到项目目录内，再写回运行时配置。

### 4. 对应配置

`configs/app.yaml` 示例：

```yaml
providers:
  tts: coqui_xtts_v2

tts:
  coqui_xtts_v2:
    model_dir: models/tts/coqui/xtts_v2
    speaker_wav_path: data/tts/coqui_xtts_v2/reference.wav
    language: zh-cn
    device: cuda:0
```

### 5. 当前限制

- Coqui XTTS v2 当前依赖 `coqui-tts[codec]`
- 该依赖在同一个 `.venv` 里可能与部分旧版 `modelscope` 依赖产生版本冲突
- 如果你计划长期同时使用 Paraformer 和 Coqui，建议优先验证完整链路
- 若后续需要更稳的部署方式，建议把 Coqui 独立到单独运行环境或单独子进程

## 模型与资源目录约定

- 头像素材：`data/avatars/<avatar_id>/`
- ASR 模型：`models/asr/...`
- sherpa TTS 模型：`models/tts/sherpa-onnx-vits-zh-ll`、`models/tts/vits-melo-tts-zh_en`
- Qwen TTS 模型：`models/tts/qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- Qwen TTS tokenizer：`models/tts/qwen/Qwen3-TTS-Tokenizer-12Hz`
- Coqui XTTS v2 模型：`models/tts/coqui/xtts_v2`
- Coqui XTTS v2 参考音频：`data/tts/coqui_xtts_v2/reference.wav`
- Wav2Lip 权重：`models/avatar/wav2lip256/wav2lip256.pth`
- 运行日志：`runtime/logs/`
- 临时文件：`runtime/tmp/`

## 下载 Qwen3-TTS 模型

模型和 tokenizer 必须下载到项目目录内，不能依赖项目目录外资源。

使用 Hugging Face：

```powershell
.\scripts\download_qwen_tts_models.ps1 -Source HuggingFace
```

使用 ModelScope：

```powershell
.\scripts\download_qwen_tts_models.ps1 -Source ModelScope
```

默认下载到：

- `models/tts/qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/`
- `models/tts/qwen/Qwen3-TTS-Tokenizer-12Hz/`

## 绿色包

绿色包不是单文件 `exe`，而是类似 ComfyUI portable 的整目录分发模式：

- `python/` 提供项目内独立 Python 运行时
- `tools/ffmpeg/bin/` 提供项目内独立 ffmpeg
- `models/`、`data/avatars/` 提供项目内模型和素材
- `start_nvidia_gpu.bat` 为绿色包默认启动入口

### 1. 准备绿色包运行时

```powershell
.\scripts\prepare_portable_runtime.ps1
```

默认行为：

- 自动优先使用 `py -3.10`
- 自动查找本机 `ffmpeg`
- 复制 Python 3.10 到项目根目录 `python/`
- 复制 ffmpeg 到项目根目录 `tools/ffmpeg/bin/`
- 安装 `requirements-torch-cu128.txt`、`requirements.txt`
- 默认不安装 Qwen / Coqui 的可选依赖
- 如需启用 Qwen3-TTS，显式传入 `-IncludeQwenTts`
- 如需启用 Coqui XTTS v2，显式传入 `-IncludeCoquiTts`

如需显式指定来源：

```powershell
.\scripts\prepare_portable_runtime.ps1 `
  -PythonExe "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe" `
  -FfmpegBinDir "D:\Program Files\ffmpeg\bin"
```

### 2. 构建绿色包

```powershell
.\scripts\build_portable_package.ps1
```

默认输出：

- `dist/LiveTalking-portable-win64-nvidia-cu128/`
- 默认不把 Qwen / Coqui 模型与参考音频打进包体
- 如需把 Qwen3-TTS 带入绿色包，显式传入 `-IncludeQwenTts`
- 如需把 Coqui XTTS v2 带入绿色包，显式传入 `-IncludeCoquiTts`

如需 zip：

```powershell
.\scripts\build_portable_package.ps1 -SkipZip:$false
```

### Qwen / Coqui 在绿色包中的行为

- 默认绿色包不包含 Qwen / Coqui 的模型、tokenizer 和参考音频
- 只有显式传入 `-IncludeQwenTts` / `-IncludeCoquiTts`，打包脚本才会尝试把对应资源带入绿色包
- 如果包体里没带入对应引擎，运行时会自动回落到 `vits_zh`
- 如果 `configs/app.yaml` 中默认 TTS 已切到 `qwen3_customvoice` 或 `coqui_xtts_v2`，但你没有显式开启对应开关，打包会直接失败
- 如果显式开启了对应开关，但项目内缺少模型或参考音频，打包也会失败

详细说明见 [docs/PORTABLE_PACKAGE.md](docs/PORTABLE_PACKAGE.md)。

## 配置说明

- 默认运行配置：`configs/app.yaml`
- 配置参考模板：`configs/app.example.yaml`
- 建议通过环境变量注入密钥：
  - `LIVETALKING_OPENAI_API_KEY`
  - `OPENAI_API_KEY`

`configs/app.yaml` 应继续使用项目内相对路径，不依赖开发机绝对路径。

## 辅助脚本

- 本地资源准备：`scripts/prepare_local_assets.ps1`
- Qwen 模型下载：`scripts/download_qwen_tts_models.ps1`
- Coqui XTTS v2 模型下载：`scripts/download_coqui_xtts_v2_models.ps1`
- 绿色运行时准备：`scripts/prepare_portable_runtime.ps1`
- 绿色包打包：`scripts/build_portable_package.ps1`
