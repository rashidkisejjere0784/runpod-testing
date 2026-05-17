import base64
import io
import wave
import torch
import runpod

from transformers import TextStreamer
from unsloth import FastLanguageModel

model = None
tokenizer = None


def load_model():
    global model, tokenizer

    if model is None:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/orpheus-3b-0.1-ft-bnb-4bit",
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
        )

        FastLanguageModel.for_inference(model)

    return model, tokenizer


def tts_handler(event):
    global model, tokenizer

    if model is None:
        model, tokenizer = load_model()

    input_data = event.get("input", {})

    text = input_data.get("text")
    voice = input_data.get("voice", "tara")

    if not text:
        return {"error": "No text provided."}

    # Orpheus prompt format
    prompt = f"<|voice:{voice}|>{text}<|audio|>"

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=1200,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
        )

    tokens = output[0]

    # Decode generated audio tokens
    # NOTE:
    # Depending on the checkpoint, you may need
    # SNAC or DAC decoder here.
    audio_values = tokenizer.decode(tokens)

    # Placeholder WAV generation
    audio_buffer = io.BytesIO()

    with wave.open(audio_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        if isinstance(audio_values, bytes):
            wf.writeframes(audio_values)
        else:
            wf.writeframes(b"")

    audio_base64 = base64.b64encode(
        audio_buffer.getvalue()
    ).decode("utf-8")

    return {
        "voice": voice,
        "audio_base64": audio_base64,
    }


runpod.serverless.start(
    {"handler": tts_handler}
)