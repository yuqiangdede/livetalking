# LiveTalking Local

本项目基于 [lipku/LiveTalking](https://github.com/lipku/LiveTalking) fork，并整理为一条可维护的本地运行链路：

`wav2lip + Paraformer + sherpa-onnx / EdgeTTS + LLM + WebRTC`

这次整理的目标不是保留上游全部能力，而是把仓库收敛到适合继续维护、公开上传和本地部署的最小可用方案。

## 当前保留范围

- 数字人渲染：`wav2lip`
- 语音识别：`Paraformer + 标点恢复`
- 语音合成：`sherpa-onnx`、`EdgeTTS`
- 对话：OpenAI-compatible LLM / LM Studio
- 交互方式：`WebRTC` 浏览器控制台

## 当前已清理

- 历史 Web 演示页和 `rtcpush` 相关页面
- 非当前主链路的 TTS 适配器逻辑
- 本地缓存、运行日志、`__pycache__`
- 带机器绝对路径的配置和脚本默认值
- 仓库内明文 API Key
- 大型模型权重改为 Git LFS 管理

## 项目结构

```text
app.py
livetalking/
configs/
data/
models/
runtime/
scripts/
tests/
wav2lip/
web/
docs/
```

结构说明见 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)。

## 运行要求

- Python `3.10`
- 项目内虚拟环境：`.venv`
- 模型、头像、临时文件全部位于项目目录内
- 默认入口为浏览器控制台 `WebRTC`
- 拉取带权重的仓库后先执行 `git lfs install` 和 `git lfs pull`

## 快速启动

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
.\.venv\Scripts\python.exe -m pip install "setuptools<81"
.\.venv\Scripts\python.exe -m pip install -r .\requirements-torch-cu128.txt
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe .\app.py
```

启动后打开：

`http://127.0.0.1:8010/`

## 配置说明

- 默认运行配置：`configs/app.yaml`
- 配置参考模板：`configs/app.example.yaml`
- 建议通过环境变量注入密钥：
  - `LIVETALKING_OPENAI_API_KEY`
  - `OPENAI_API_KEY`

`configs/app.yaml` 已清理为可公开提交的安全默认值，不再包含本地私有密钥。

## 资源目录约定

- 头像素材：`data/avatars/<avatar_id>/`
- ASR 模型：`models/asr/...`
- TTS 模型：`models/tts/...`
- Wav2Lip 权重：`models/avatar/wav2lip256/wav2lip256.pth`
- 运行日志：`runtime/logs/`
- 临时文件：`runtime/tmp/`

当前仓库中的 `.pt`、`.pth`、`.onnx` 文件由 Git LFS 承载，克隆后需要执行 `git lfs pull` 才能拿到真实权重。
头像素材仍按普通 Git 文件管理；如果后续体积继续增长，可以再按同样方式迁移到 LFS。

## 辅助脚本

- 资源准备：`scripts/prepare_local_assets.ps1`
- 单元测试：`tests/`

## 主要接口

- `POST /offer`
- `POST /human`
- `POST /humanaudio`
- `POST /api/asr/transcribe`
- `POST /api/dialog/add`
- `GET /api/dialog/history`
- `POST /api/dialog/clear`
- `POST /set_silence_gate`
- `POST /record`
- `POST /interrupt_talk`
- `POST /is_speaking`
- `GET /api/runtime/config`
- `POST /api/runtime/config`
