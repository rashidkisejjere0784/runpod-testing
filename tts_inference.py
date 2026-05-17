import base64
import io
import os
import tempfile
import torch
import runpod

from TTS.api import TTS

model = None


def load_model():
    global model

    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    return model


def tts_handler(event):
    global model

    if model is None:
        load_model()

    input_data = event.get("input", {})
    text = input_data.get("text")
    language = input_data.get("language", "en")
    speaker = input_data.get("speaker", "Claribel Dervla")

    if not text:
        return {"error": "No text provided."}

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        model.tts_to_file(
            text=text,
            speaker=speaker,
            language=language,
            file_path=tmp_path,
        )

        with open(tmp_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

    finally:
        os.unlink(tmp_path)

    return {
        "speaker": speaker,
        "language": language,
        "audio_base64": audio_base64,
    }

runpod.serverless.start({"handler": tts_handler})