import os
from cached_path import cached_path
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import load_model, load_vocoder

MODELS = {
    "Unified": ("hf://SWivid/Habibi-TTS/Unified/vocab.txt", "hf://SWivid/Habibi-TTS/Unified/model_200000.safetensors"),
    "SAU":     ("hf://SWivid/Habibi-TTS/Specialized/SAU/vocab.txt", "hf://SWivid/Habibi-TTS/Specialized/SAU/model_200000.safetensors"),
    "EGY":     ("hf://SWivid/Habibi-TTS/Specialized/EGY/vocab.txt", "hf://SWivid/Habibi-TTS/Specialized/EGY/model_100000.safetensors"),
    "MSA":     ("hf://SWivid/Habibi-TTS/Specialized/MSA/vocab.txt", "hf://SWivid/Habibi-TTS/Specialized/MSA/model_200000.safetensors"),
}
model_name = os.environ.get("HABIBI_MODEL", "Unified")
cfg = MODELS.get(model_name, MODELS["Unified"])
vocab = str(cached_path(cfg[0]))
ckpt  = str(cached_path(cfg[1]))
base  = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
load_model(DiT, base, ckpt, vocab_file=vocab)
load_vocoder()
print(f"Model {model_name} pre-downloaded OK")