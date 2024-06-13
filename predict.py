from typing import Optional
from scipy.io.wavfile import write as write_wav
from cog import BasePredictor, Input, Path, BaseModel
from bark import SAMPLE_RATE, generate_audio, preload_models, save_as_prompt
from bark.api import semantic_to_waveform
from bark.generation import ALLOWED_PROMPTS, generate_text_semantic
import numpy as np
import os


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
        )
    ) -> Path:
        """Run a single prediction on the model"""

        prompt = prompt.replace("\n", " ").strip()
        sentences = [sent.strip() + "." if not sent.strip().endswith(".") else sent.strip() for sent in prompt.split(". ")]
        silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter of silence

        pieces = []
        for sentence in sentences:
            semantic_tokens = generate_text_semantic(
                sentence,
                history_prompt=history_prompt,
                min_eos_p=0.05
            )

            audio_array = semantic_to_waveform(semantic_tokens, history_prompt=history_prompt)

            pieces += [audio_array, silence.copy()]

        audio = np.concatenate(pieces)

        output = "/tmp/audio.wav"
        final_output = "/tmp/audio.mp3"
        write_wav(output, SAMPLE_RATE, audio)
        os.system(f"ffmpeg -i {output} -codec:a libmp3lame -q:a 2 {final_output}")
        return Path(final_output)
