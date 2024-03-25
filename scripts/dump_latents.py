import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from pflow_encodec.data.tokenizer import EncodecTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_tsv", type=str, required=True)
    parser.add_argument("--output_ext", type=str, help="output csv file", default=".latent.npy")

    args = parser.parse_args()

    df = pd.read_csv(args.input_tsv, sep="\t", engine="pyarrow")
    tokenizer = EncodecTokenizer()
    paths = df["audio_path"].tolist()
    with torch.inference_mode():
        for path in tqdm(paths):
            output_path = Path(path).with_suffix(args.output_ext)
            if output_path.exists():
                continue
            latent = tokenizer.encode_file(path, return_code=False)
            np.save(output_path, latent.cpu().numpy().astype(np.float32))
