import math
import tempfile
import unittest
import wave
from pathlib import Path


def _make_wav_bytes(*, sr: int = 16000, seconds: float = 1.0, freq: float = 440.0) -> bytes:
    n = int(sr * seconds)
    frames = bytearray()
    for i in range(n):
        s = math.sin(2.0 * math.pi * freq * (i / sr))
        v = int(max(-1.0, min(1.0, s)) * 32767.0)
        frames += int(v).to_bytes(2, byteorder="little", signed=True)
    with tempfile.TemporaryFile() as f:
        with wave.open(f, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(bytes(frames))
        f.seek(0)
        return f.read()


class TestVoiceLibrary(unittest.TestCase):
    def test_create_list_get_delete(self):
        from hub.voice_library import VoiceLibrary

        with tempfile.TemporaryDirectory() as td:
            hub_root = Path(td)
            lib = VoiceLibrary(hub_root=hub_root)

            wav_bytes = _make_wav_bytes()
            meta = lib.create_voice(name="Test Voice", input_bytes=wav_bytes, filename="sample.wav")
            voice_id = meta["id"]

            # Files exist
            voice_dir = hub_root / "outputs" / "voices" / voice_id
            self.assertTrue((voice_dir / "meta.json").exists())
            self.assertTrue((voice_dir / "prompt.wav").exists())

            # List
            voices = lib.list_voices()
            self.assertTrue(any(v.id == voice_id for v in voices))

            # Get meta + audio
            got = lib.get_voice_meta(voice_id)
            self.assertEqual(got["id"], voice_id)
            wav_path = lib.get_voice_audio_path(voice_id)
            self.assertTrue(wav_path.exists())

            # Delete
            lib.delete_voice(voice_id)
            self.assertFalse(voice_dir.exists())

