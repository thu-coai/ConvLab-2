# Speech Recognition

A TTS+ASR pipeline to simulate speech characteristics and recognition error.

## TTS

We use gTTS as the TTS moudle.
Pleas install ffmpeg before use:
```bash
conda install ffmpeg
```

## ASR

We use DeepSpeech as the ASR moudle. Noting that we use DeepSpeech2 to conduct our experiments in our paper, but in this released toolkit we choose DeepSpeech instead for higher efficiency.

Please download [released models](https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3) before use.
Please download deepspeech-0.9.3-models.pbmm and deepspeech-0.9.3-models.scorer place them under `Speech Recognition/` dir.
