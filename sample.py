# make_samples.py
from gtts import gTTS
import subprocess

samples = {
    "phish": "Hello, this is Priya from Acme Bank's fraud team. We detected suspicious activity on your account. Please verify your identity by reading the one-time password we just sent to your phone. What is the OTP now?",
    "normal": "Hey Raj, are you coming to the college canteen at 2 PM? I’ll bring the notes and we’ll revise the assignment together. See you there."
}

for name, text in samples.items():
    mp3 = f"{name}.mp3"
    wav = f"{name}.wav"
    tts = gTTS(text=text, lang='en')
    tts.save(mp3)
    # convert mp3 to 16kHz mono WAV (whisper prefers 16kHz, but openai-whisper handles resampling too)
    