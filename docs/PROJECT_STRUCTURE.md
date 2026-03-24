# 项目结构

## 目标

这个仓库现在只围绕一条可维护的本地运行链路组织：

`wav2lip + Paraformer + sherpa-onnx / EdgeTTS + LLM + WebRTC`

不再以保留上游全部演示分支为目标。

## 根目录

```text
app.py
configs/
data/
docs/
livetalking/
models/
runtime/
scripts/
tests/
wav2lip/
web/
```

## 目录职责

- `app.py`
  - 根目录兼容启动入口，内部转到 `livetalking.app`
- `configs/`
  - 可提交的运行配置和示例配置
- `data/`
  - 头像素材目录，默认使用 `data/avatars/<avatar_id>/`
- `docs/`
  - 项目说明文档
- `livetalking/`
  - 当前实际运行时代码
- `models/`
  - ASR / TTS / Wav2Lip 模型与权重
- `runtime/`
  - 日志和临时文件
- `scripts/`
  - 项目内资源准备脚本
- `tests/`
  - 单元测试
- `wav2lip/`
  - 项目运行依赖的 Wav2Lip 代码
- `web/`
  - 浏览器控制台页面和前端脚本

## Python 包结构

```text
livetalking/
  app.py
  avatar/
  config/
  core/
  providers/
  realtime/
  utils/
```

### 关键模块

- `livetalking/app.py`
  - 进程启动、配置加载、WebRTC 接口、静态资源挂载
- `livetalking/config/app_config.py`
  - 配置模型、路径解析、配置读写
- `livetalking/avatar/wav2lip_real.py`
  - Wav2Lip 口型驱动与视频帧输出
- `livetalking/providers/local_asr.py`
  - Paraformer ASR、音频解码、热词与近音词处理
- `livetalking/providers/sherpa_tts.py`
  - Sherpa-ONNX TTS
- `livetalking/providers/tts_engines.py`
  - 当前只保留 EdgeTTS 的轻量适配层
- `livetalking/providers/llm_client.py`
  - OpenAI-compatible / LM Studio LLM 调用
- `livetalking/realtime/webrtc.py`
  - WebRTC 播放轨道封装

## 资源约定

- Wav2Lip 权重：`models/avatar/wav2lip256/wav2lip256.pth`
- Paraformer：`models/asr/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
- 标点模型：`models/asr/punc_ct-transformer_zh-cn-common-vocab272727-pytorch`
- Sherpa 中文：`models/tts/sherpa-onnx-vits-zh-ll`
- Sherpa 中英：`models/tts/vits-melo-tts-zh_en`
- 头像：`data/avatars/<avatar_id>/`
- 真实模型和头像资源默认由用户在项目目录内本地准备，不随仓库正文上传

## 前端入口

- `web/index.html`
  - 默认入口，跳转到当前主控台页面
- `web/webrtcapi-asr.html`
  - 当前唯一保留的主控台页面
- `web/client.js`
  - WebRTC 协商逻辑
