import base64
import io
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
    speaker = input_data.get("speaker", "Claribel Dervla")  # default XTTS-v2 speaker

    if not text:
        return {"error": "No text provided."}

    audio_buffer = io.BytesIO()

    model.tts_to_file(
        text=text,
        speaker=speaker,
        language=language,
        file_path=audio_buffer,
        pipe_out=True,
    )

    audio_base64 = base64.b64encode(
        audio_buffer.getvalue()
    ).decode("utf-8")

    return {
        "speaker": speaker,
        "language": language,
        "audio_base64": audio_base64,
    }


runpod.serverless.start(
    {"handler": tts_handler}
)