import torch
import logging
from kokoro import KPipeline
import sounddevice as sd
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize TTS pipeline
pipeline = KPipeline(lang_code='a',device='cpu')  # Adjust lang_code as needed

def text_to_audio_speaker(text, sample_rate=24000 , voice = 'af_bella'):
    """Generate audio for Speaker 1 (teacher voice)"""
    audios = []
    generator = pipeline(text, voice=voice)
    for _, (_, _, audio) in enumerate(generator):
        audios.append(audio)
    audio_tensor = torch.cat(audios, dim=0)
    audio_np = audio_tensor.cpu().numpy()
    sd.play(audio_np, samplerate=sample_rate)
    sd.wait()

kokoro_voices = {
    'a': {  # American English
        'female': [
            'af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica',
            'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky'
        ],
        'male': [
            'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam',
            'am_michael', 'am_onyx', 'am_puck'
        ]
    },
    'b': {  # British English
        'female': [
            'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily'
        ],
        'male': [
            'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis'
        ]
    },
    'f': {  # French
        'female': ['ff_siwis'],
        'male': []
    },
    'i': {  # Italian
        'female': ['if_sara'],
        'male': ['im_nicola']
    },
    'j': {  # Japanese
        'female': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro'],
        'male': ['jm_kumo']
    },
    'z': {  # Chinese (Mandarin)
        'female': ['zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi'],
        'male': ['zm_yunjian', 'zm_yunxi', 'zm_yunxia', 'zm_yunyang']
    },
}

# for lang, genders in kokoro_voices.items():
#     for gender, voices in genders.items():
#         for voice in voices:
#             print(f"{lang} - {gender}: {voice}")
# text_to_audio_speaker("You know, there's something about the way you look at the stars... like you're trying to understand the universe", voice = "af_aoede")

# af_aoede
# af_bella
# af_heart
# af_nicole r
# af_sarah  r
# am_puck  r b
# zm_yunyang r b
# am_eric b
