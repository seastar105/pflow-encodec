import argparse
from pathlib import Path

import pandas as pd
import torch
from audiotools import AudioSignal
from tqdm.auto import tqdm


class AudioOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, input_path_list, output_sr):
        super().__init__()
        self.input_path_list = input_path_list
        self.output_sr = output_sr

    def __len__(self):
        return len(self.input_path_list)

    def __getitem__(self, idx):
        try:
            input_path = self.input_path_list[idx]
            signal = AudioSignal(input_path)
            if signal.sample_rate != self.output_sr:
                signal = signal.resample(self.output_sr)
            if signal.num_channels > 1:
                signal = signal.to_mono()
            signal = signal.ensure_max_of_audio()
            duration = signal.duration
        except Exception as e:
            print(f"Error in {input_path}")
            return None, None

        return signal, duration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path_list", type=str, required=True)
    parser.add_argument("--output_path_list", type=str, required=True)
    parser.add_argument("--output_sr", type=int, default=24000)
    parser.add_argument("--output_tsv_path", type=str, required=True)

    arguments = parser.parse_args()
    with open(arguments.input_path_list) as f:
        input_path_list = [line.strip() for line in f]
    with open(arguments.output_path_list) as f:
        output_path_list = [line.strip() for line in f]

    ds = AudioOnlyDataset(input_path_list, arguments.output_sr)
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=32)
    durations = []
    for i, (signal, duration) in tqdm(enumerate(dl), total=len(dl)):
        output_path = output_path_list[i]
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            if not Path(output_path).exists():
                signal.write(output_path)
            durations.append(duration)
        except Exception as e:
            print(f"Error in {output_path}")

    df = pd.DataFrame({"audio_path": output_path_list, "duration": durations})
    df.to_csv(arguments.output_tsv_path, sep="\t", index=False)
