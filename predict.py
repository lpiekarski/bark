from typing import Optional
from scipy.io.wavfile import write as write_wav
from cog import BasePredictor, Input, Path, BaseModel
from bark import SAMPLE_RATE, generate_audio, preload_models, save_as_prompt
from bark.generation import ALLOWED_PROMPTS


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # for the pushed version on Replicate, the CACHE_DIR from bark/generation.py is changed to a local folder to
        # include the weights file in the image for faster inference
        preload_models()

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests "
            "such as playing tic tac toe.",
        ),
        history_prompt: str = Input(
            description="history choice for audio cloning, choose from the list",
            default=None,
            choices=sorted(list(ALLOWED_PROMPTS)),
        ),
        text_temp: float = Input(
            description="generation temperature (1.0 more diverse, 0.0 more conservative)",
            default=0.7,
        ),
        waveform_temp: float = Input(
            description="generation temperature (1.0 more diverse, 0.0 more conservative)",
            default=0.7,
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        prompt = prompt.replace("\n", " ").strip()
        sentences = [sent.strip() + "." if not sent.strip().endswith(".") else sent.strip() for sent in prompt.split(". ")]
        silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter of silence

        pieces = []
        for sentence in sentences:
            audio_array = generate_audio(
                sentence,
                history_prompt=history_prompt,
                text_temp=text_temp,
                waveform_temp=waveform_temp,
            )
            pieces += [audio_array, silence.copy()]

        audio = np.concatenate(pieces)

        output = "/tmp/audio.wav"
        write_wav(output, SAMPLE_RATE, audio)
        return Path(output)
