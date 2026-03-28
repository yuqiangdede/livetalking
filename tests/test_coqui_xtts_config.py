from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from livetalking.config.app_config import (
    apply_config_to_opt,
    get_settings_db_path,
    load_app_config,
    normalize_tts_engine,
    save_app_config,
)


class CoquiXttsConfigTests(unittest.TestCase):
    def test_normalize_tts_engine_supports_coqui_aliases(self) -> None:
        self.assertEqual(normalize_tts_engine("coqui_tts"), "coqui_xtts_v2")
        self.assertEqual(normalize_tts_engine("coqui_xtts_v2"), "coqui_xtts_v2")
        self.assertEqual(normalize_tts_engine("xtts"), "coqui_xtts_v2")
        self.assertEqual(normalize_tts_engine("xtts_v2"), "coqui_xtts_v2")

    def test_load_app_config_reads_coqui_section(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "providers": {"asr": "paraformer", "tts": "coqui_xtts_v2", "avatar": "wav2lip", "llm": "lm_studio_api"},
                "runtime": {"temp_dir": "runtime/tmp", "log_level": "INFO"},
                "asr": {"paraformer": {}},
                "tts": {
                    "vits_zh": {},
                    "vits_melo_zh_en": {},
                    "qwen3_customvoice": {},
                    "coqui_xtts_v2": {
                        "model_dir": "models/tts/coqui/xtts_v2",
                        "speaker_wav_path": "data/tts/coqui_xtts_v2/reference.wav",
                        "language": "en",
                        "device": "cuda:0",
                        "speed": 1.3,
                    },
                    "edgetts": {},
                    "pyttsx3": {},
                },
                "avatar": {"wav2lip": {}},
                "llm": {"openai_chat": {}, "chat_completions_api": {}, "lm_studio_api": {}},
            }
            (config_dir / "app.yaml").write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")

            config = load_app_config(project_root=root)

            self.assertEqual(config.providers.tts, "coqui_xtts_v2")
            self.assertEqual(config.tts.coqui_xtts_v2.model_dir, "models/tts/coqui/xtts_v2")
            self.assertEqual(config.tts.coqui_xtts_v2.speaker_wav_path, "data/tts/coqui_xtts_v2/reference.wav")
            self.assertEqual(config.tts.coqui_xtts_v2.language, "en")
            self.assertEqual(config.tts.coqui_xtts_v2.device, "cuda:0")
            self.assertAlmostEqual(config.tts.coqui_xtts_v2.speed, 1.3)

    def test_save_app_config_persists_coqui_runtime_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_app_config(project_root=root)
            config.providers.tts = "coqui_xtts_v2"
            config.tts.coqui_xtts_v2.model_dir = "models/tts/coqui/xtts_v2"
            config.tts.coqui_xtts_v2.speaker_wav_path = "data/tts/coqui_xtts_v2/reference.wav"
            config.tts.coqui_xtts_v2.language = "zh-cn"
            config.tts.coqui_xtts_v2.device = "cpu"
            config.tts.coqui_xtts_v2.speed = 1.25

            db_path = Path(save_app_config(config))

            self.assertEqual(db_path, get_settings_db_path(root))
            reloaded = load_app_config(project_root=root)
            self.assertEqual(reloaded.providers.tts, "coqui_xtts_v2")
            self.assertEqual(reloaded.tts.coqui_xtts_v2.model_dir, "models/tts/coqui/xtts_v2")
            self.assertEqual(reloaded.tts.coqui_xtts_v2.speaker_wav_path, "data/tts/coqui_xtts_v2/reference.wav")
            self.assertEqual(reloaded.tts.coqui_xtts_v2.language, "zh-cn")
            self.assertEqual(reloaded.tts.coqui_xtts_v2.device, "cpu")
            self.assertAlmostEqual(reloaded.tts.coqui_xtts_v2.speed, 1.25)

    def test_apply_config_to_opt_sets_coqui_runtime_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_app_config(project_root=root)
            config.providers.tts = "coqui_xtts_v2"
            config.tts.coqui_xtts_v2.model_dir = "models/tts/coqui/xtts_v2"
            config.tts.coqui_xtts_v2.speaker_wav_path = "data/tts/coqui_xtts_v2/reference.wav"
            config.tts.coqui_xtts_v2.language = "en"
            config.tts.coqui_xtts_v2.device = "cpu"
            config.tts.coqui_xtts_v2.speed = 1.4

            opt = type("Opt", (), {})()
            opt.fps = 50

            apply_config_to_opt(opt, config)

            self.assertEqual(opt.tts, "coqui_xtts_v2")
            self.assertEqual(opt.REF_FILE, str(root / "data/tts/coqui_xtts_v2/reference.wav"))
            self.assertEqual(opt.TTS_MODEL_DIR, str(root / "models/tts/coqui/xtts_v2"))
            self.assertEqual(opt.TTS_SPEAKER_WAV_PATH, str(root / "data/tts/coqui_xtts_v2/reference.wav"))
            self.assertEqual(opt.TTS_LANGUAGE, "en")
            self.assertEqual(opt.TTS_DEVICE, "cpu")
            self.assertAlmostEqual(opt.TTS_SPEED, 1.4)
            self.assertEqual(opt.TTS_PROVIDER, "")

    def test_apply_config_to_opt_falls_back_to_existing_mp3_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data/tts/coqui_xtts_v2").mkdir(parents=True, exist_ok=True)
            (root / "data/tts/coqui_xtts_v2/reference.mp3").write_bytes(b"demo")

            config = load_app_config(project_root=root)
            config.providers.tts = "coqui_xtts_v2"
            config.tts.coqui_xtts_v2.model_dir = "models/tts/coqui/xtts_v2"
            config.tts.coqui_xtts_v2.speaker_wav_path = "data/tts/coqui_xtts_v2/reference.wav"
            config.tts.coqui_xtts_v2.language = "zh-cn"
            config.tts.coqui_xtts_v2.device = "cpu"
            config.tts.coqui_xtts_v2.speed = 0.95

            opt = type("Opt", (), {})()
            opt.fps = 50

            apply_config_to_opt(opt, config)

            self.assertEqual(config.tts.coqui_xtts_v2.speaker_wav_path, "data/tts/coqui_xtts_v2/reference.mp3")
            self.assertEqual(opt.REF_FILE, str(root / "data/tts/coqui_xtts_v2/reference.mp3"))
            self.assertEqual(opt.TTS_SPEAKER_WAV_PATH, str(root / "data/tts/coqui_xtts_v2/reference.mp3"))
            self.assertAlmostEqual(opt.TTS_SPEED, 0.95)


if __name__ == "__main__":
    unittest.main()
