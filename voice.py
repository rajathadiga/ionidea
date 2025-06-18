import sounddevice as sd
import numpy as np
import time
import queue
import threading
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class SpeechTranscriber:
    def __init__(self, model_id="openai/whisper-tiny", language="english", sample_rate=16000, auto_calibrate=True):
        self.model_id = model_id
        self.language = language
        self.sample_rate = sample_rate
        self.chunk_duration = 0.5
        self.chunk_size = int(sample_rate * self.chunk_duration)
        self.device = torch.device("cpu")  # 🔁 Force CPU

        print("🔧 Loading Whisper model and processor on CPU...")
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(self.device)
        self.decoder_prompt_ids = self.processor.get_decoder_prompt_ids(language=self.language, task="transcribe")

        self.threshold = self.calibrate_threshold() if auto_calibrate else 0.02

    def calibrate_threshold(self, duration=1.0, base_threshold=0.02):
        print("🔈 Calibrating background noise... Please remain silent.")
        noise_levels = []
        with sd.InputStream(samplerate=self.sample_rate, channels=1, blocksize=self.chunk_size) as stream:
            for _ in range(int(duration / self.chunk_duration)):
                audio_data, _ = stream.read(self.chunk_size)
                noise_levels.append(np.mean(np.abs(audio_data)))
        avg_noise = np.mean(noise_levels)
        calibrated = max(avg_noise * 3, base_threshold)
        print(f"✅ Calibration complete. Threshold: {calibrated:.6f}")
        return calibrated

    def record_until_pause(self, threshold, speech_timeout=1.5, min_speech_duration=0.3):
        audio_queue = queue.Queue()
        stop_recording = threading.Event()

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"⚠️ Audio status: {status}")
            audio_queue.put(indata.copy())

        def has_speech(chunk):
            return np.mean(np.abs(chunk)) > threshold

        recorded_audio = []
        speech_detected = False
        silence_start_time = None
        consecutive_speech_chunks = 0
        min_chunks = int(min_speech_duration / self.chunk_duration)

        print("🎤 Listening... Speak now.")
        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=audio_callback, blocksize=self.chunk_size):
            while not stop_recording.is_set():
                try:
                    chunk = audio_queue.get(timeout=1.0).flatten()
                    if has_speech(chunk):
                        consecutive_speech_chunks += 1
                        if consecutive_speech_chunks >= min_chunks and not speech_detected:
                            print("🎙️ Speech detected. Recording...")
                            speech_detected = True
                        silence_start_time = None
                        if speech_detected:
                            recorded_audio.append(chunk)
                    else:
                        consecutive_speech_chunks = 0
                        if speech_detected:
                            if silence_start_time is None:
                                silence_start_time = time.time()
                            recorded_audio.append(chunk)
                            if time.time() - silence_start_time > speech_timeout:
                                stop_recording.set()
                except queue.Empty:
                    continue

        return np.concatenate(recorded_audio) if recorded_audio else None

    def transcribe(self, audio_array):
        if audio_array is None or len(audio_array) == 0:
            return ""

        inputs = self.processor(
            audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).to(self.device)

        prediction = self.model.generate(
            inputs.input_features,
            attention_mask=inputs.get("attention_mask", None),
            forced_decoder_ids=self.decoder_prompt_ids
        )
        return self.processor.batch_decode(prediction, skip_special_tokens=True)[0].strip()

    def record_and_transcribe(self):
        audio_data = self.record_until_pause(self.threshold)
        return self.transcribe(audio_data)

# Example usage
if __name__ == "__main__":
    transcriber = SpeechTranscriber(model_id="openai/whisper-tiny")
    try:
        result = transcriber.record_and_transcribe()
        if result:
            print(f"\n🗣️ Transcription: {result}")
        else:
            print("🤐 No speech detected.")
    except KeyboardInterrupt:
        print("\n🛑 Manually interrupted.")
    except Exception as e:
        print(f"❌ Error: {e}")
