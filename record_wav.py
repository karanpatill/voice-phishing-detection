# record_wav.py
import sounddevice as sd
import soundfile as sf

DURATION = 10  # seconds
SR = 16000     # sample rate
FNAME = "sample.wav"

print("Recording for", DURATION, "seconds...")
audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='int16')
sd.wait()
sf.write(FNAME, audio, SR, subtype='PCM_16')
print("Saved:", FNAME)
