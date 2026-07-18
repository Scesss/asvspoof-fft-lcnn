import torch
from torch import nn


class FFTTransform(nn.Module):
    def __init__(self):
        super().__init__()

        window = torch.blackman_window(1724)
        self.register_buffer("window", window)

    def forward(self, audio):
        audio = audio.squeeze(1)
        audio = torch.stft(
            input=audio,
            n_fft=1724,
            hop_length=128,
            win_length=1724,
            window=self.window,
            center=False,
            return_complex=True,
        )
        log_power = torch.log(audio.abs().pow(2) + 1e-6)
        log_power = log_power.unsqueeze(1)
        return log_power
