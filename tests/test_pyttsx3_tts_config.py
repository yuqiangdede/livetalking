from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from livetalking.config.app_config import (
    apply_config_to_opt,
    get_settings_db_path,
    load_app_config,
    normalize_tts_engine,
    save_app_config,
)
from livetalking.providers import pyttsx3_tts


class Pyttsx3TtsConfigTests(unittest.TestCase):
    def test_list_pyttsx3_voice_options_is_limited_to_fixed_choices(self) -> None:
        class DummyVoice:
            def __init__(self, voice_id: str, name: str) -> None:
                self.id = voice_id
                self.name = name

        class DummyEngine:
            def getProperty(self, name: str):
                if name != "voices":
                    return []
                return [
                    DummyVoice(
                        "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_ZH-CN_HUIHUI_11.0",
                        "Microsoft Huihui Desktop - Chinese (Simplified)",
                    ),
                    DummyVoice(
                        "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0",
                        "Microsoft Zira Desktop - English (United States)",
                    ),
                    DummyVoice("demo.voice", "Unexpected Voice"),
                ]

            def stop(self) -> None:
                return None

        with patch.object(pyttsx3_tts, "pyttsx3") as fake_module:
            fake_module.init.return_value = DummyEngine()
            options = pyttsx3_tts.list_pyttsx3_voice_options()

        self.assertEqual(
            options,
            [
                {
                    "id": "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_ZH-CN_HUIHUI_11.0",
                    "label": "Huihui（中文）",
                    "desc": "Microsoft Huihui Desktop - Chinese (Simplified) / HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_ZH-CN_HUIHUI_11.0",
                },
                {
                    "id": "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0",
                    "label": "Zira（英文）",
                    "desc": "Microsoft Zira Desktop - English (United States) / HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0",
                },
            ],
        )

    def test_normalize_tts_engine_supports_pyttsx3_aliases(self) -> None:
        self.assertEqual(normalize_tts_engine("pyttsx3"), "pyttsx3")
        self.assertEqual(normalize_tts_engine("pyttsx3-tts"), "pyttsx3")
        self.assertEqual(normalize_tts_engine("sapi5"), "pyttsx3")

    def test_load_app_config_reads_pyttsx3_section(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "providers": {"asr": "paraformer", "tts": "pyttsx3", "avatar": "wav2lip", "llm": "lm_studio_api"},
                "runtime": {"temp_dir": "runtime/tmp", "log_level": "INFO"},
                "asr": {"paraformer": {}},
                "tts": {
                    "vits_zh": {},
                    "vits_melo_zh_en": {},
                    "qwen3_customvoice": {},
                    "edgetts": {},
                    "pyttsx3": {
                        "voice_id": "demo.voice",
                        "rate": 190,
                        "volume": 0.8,
                        "driver_name": "sapi5",
                    },
                },
                "avatar": {"wav2lip": {}},
                "llm": {"openai_chat": {}, "chat_completions_api": {}, "lm_studio_api": {}},
            }
            (config_dir / "app.yaml").write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")

            config = load_app_config(project_root=root)

            self.assertEqual(config.providers.tts, "pyttsx3")
            self.assertEqual(config.tts.pyttsx3.voice_id, "demo.voice")
            self.assertEqual(config.tts.pyttsx3.rate, 190)
            self.assertAlmostEqual(config.tts.pyttsx3.volume, 0.8)
            self.assertEqual(config.tts.pyttsx3.driver_name, "sapi5")

    def test_save_app_config_persists_pyttsx3_runtime_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_app_config(project_root=root)
            config.providers.tts = "pyttsx3"
            config.tts.pyttsx3.voice_id = "demo.voice"
            config.tts.pyttsx3.rate = 210
            config.tts.pyttsx3.volume = 0.75
            config.tts.pyttsx3.driver_name = "sapi5"

            db_path = Path(save_app_config(config))

            self.assertEqual(db_path, get_settings_db_path(root))
            reloaded = load_app_config(project_root=root)
            self.assertEqual(reloaded.providers.tts, "pyttsx3")
            self.assertEqual(reloaded.tts.pyttsx3.voice_id, "demo.voice")
            self.assertEqual(reloaded.tts.pyttsx3.rate, 210)
            self.assertAlmostEqual(reloaded.tts.pyttsx3.volume, 0.75)
            self.assertEqual(reloaded.tts.pyttsx3.driver_name, "sapi5")

    def test_apply_config_to_opt_sets_pyttsx3_runtime_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_app_config(project_root=root)
            config.providers.tts = "pyttsx3"
            config.tts.pyttsx3.voice_id = "demo.voice"
            config.tts.pyttsx3.rate = 200
            config.tts.pyttsx3.volume = 0.9
            config.tts.pyttsx3.driver_name = "sapi5"

            opt = type("Opt", (), {})()
            opt.fps = 50

            apply_config_to_opt(opt, config)

            self.assertEqual(opt.tts, "pyttsx3")
            self.assertEqual(opt.REF_FILE, "")
            self.assertEqual(opt.TTS_PYTTSX3_VOICE_ID, "")
            self.assertEqual(opt.TTS_PYTTSX3_RATE, 200)
            self.assertAlmostEqual(opt.TTS_PYTTSX3_VOLUME, 0.9)
            self.assertEqual(opt.TTS_PYTTSX3_DRIVER_NAME, "sapi5")


if __name__ == "__main__":
    unittest.main()
