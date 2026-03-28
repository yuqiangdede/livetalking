from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from livetalking.config.app_config import (
    get_settings_db_path,
    load_app_config,
    normalize_tts_engine,
    save_app_config,
)
from livetalking.providers.qwen_tts import QwenCustomVoiceTTS


class QwenTtsConfigTests(unittest.TestCase):
    def test_normalize_tts_engine_supports_qwen_aliases(self) -> None:
        self.assertEqual(normalize_tts_engine("qwen3_customvoice"), "qwen3_customvoice")
        self.assertEqual(normalize_tts_engine("qwen3-tts"), "qwen3_customvoice")
        self.assertEqual(normalize_tts_engine("qwen_customvoice"), "qwen3_customvoice")

    def test_load_app_config_reads_qwen_section(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "providers": {"asr": "paraformer", "tts": "qwen3_customvoice", "avatar": "wav2lip", "llm": "lm_studio_api"},
                "runtime": {"temp_dir": "runtime/tmp", "log_level": "INFO"},
                "asr": {"paraformer": {}},
                "tts": {
                    "vits_zh": {},
                    "vits_melo_zh_en": {},
                    "qwen3_customvoice": {
                        "model_dir": "models/tts/qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                        "tokenizer_dir": "models/tts/qwen/Qwen3-TTS-Tokenizer-12Hz",
                        "device": "cuda:0",
                        "dtype": "float16",
                        "attn_implementation": "sdpa",
                        "speaker": "Ryan",
                        "instruct": "Use a calm tone.",
                        "language": "English",
                        "speed": 1.2,
                    },
                    "edgetts": {},
                },
                "avatar": {"wav2lip": {}},
                "llm": {"openai_chat": {}, "chat_completions_api": {}, "lm_studio_api": {}},
            }
            (config_dir / "app.yaml").write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")

            config = load_app_config(project_root=root)

            self.assertEqual(config.providers.tts, "qwen3_customvoice")
            self.assertEqual(config.tts.qwen3_customvoice.speaker, "Ryan")
            self.assertEqual(config.tts.qwen3_customvoice.language, "English")
            self.assertEqual(config.tts.qwen3_customvoice.speed, 1.2)

    def test_save_app_config_persists_qwen_runtime_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_app_config(project_root=root)
            config.providers.tts = "qwen3_customvoice"
            config.tts.qwen3_customvoice.speaker = "Vivian"
            config.tts.qwen3_customvoice.instruct = "用温柔的语气说"
            config.tts.qwen3_customvoice.speed = 0.9

            db_path = Path(save_app_config(config))

            self.assertEqual(db_path, get_settings_db_path(root))
            reloaded = load_app_config(project_root=root)
            self.assertEqual(reloaded.providers.tts, "qwen3_customvoice")
            self.assertEqual(reloaded.tts.qwen3_customvoice.speaker, "Vivian")
            self.assertEqual(reloaded.tts.qwen3_customvoice.instruct, "用温柔的语气说")
            self.assertEqual(reloaded.tts.qwen3_customvoice.speed, 0.9)

    def test_qwen_language_inference_uses_speaker_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_app_config(project_root=root)
            config.providers.tts = "qwen3_customvoice"
            config.tts.qwen3_customvoice.speaker = "Aiden"
            config.tts.qwen3_customvoice.language = "Auto"
            opt = type("Opt", (), {})()
            opt.fps = 50
            opt.TTS_MODEL_DIR = str(root / "models")
            opt.TTS_TOKENIZER_DIR = str(root / "tokenizer")
            opt.TTS_DEVICE = "cuda:0"
            opt.TTS_DTYPE = "float16"
            opt.TTS_ATTN_IMPLEMENTATION = "sdpa"
            opt.TTS_QWEN_SPEAKER = "Aiden"
            opt.TTS_QWEN_INSTRUCT = ""
            opt.TTS_QWEN_LANGUAGE = "Auto"
            provider = QwenCustomVoiceTTS.__new__(QwenCustomVoiceTTS)
            provider.opt = opt
            provider.parent = None
            provider.language = "Auto"
            self.assertEqual(provider._infer_language("Aiden"), "English")


if __name__ == "__main__":
    unittest.main()
