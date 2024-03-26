import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from seamless_communication.models.aligner.alignment_extractor import AlignmentExtractor
from tqdm.auto import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_tsv", type=str, required=True)
    parser.add_argument(
        "--output_ext", type=str, help="output extension of character duration", default=".duration.npy"
    )
    parser.add_argument("--empty_cache_rate", type=int, default=5000)
    args = parser.parse_args()

    extractor = AlignmentExtractor(
        aligner_model_name_or_card="nar_t2u_aligner",
        unit_extractor_model_name_or_card="xlsr2_1b_v2",
        unit_extractor_output_layer=35,
        unit_extractor_kmeans_model_uri="https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy",
        device=torch.device("cuda"),
        dtype=torch.float16,
    )

    df = pd.read_csv(args.input_tsv, sep="\t", engine="pyarrow")
    paths = df["audio_path"].tolist()
    texts = df["text"].tolist()
    errors = []
    with torch.inference_mode():
        for idx, (path, text) in tqdm(enumerate(zip(paths, texts)), total=len(paths)):
            if args.empty_cache_rate > 0 and idx % args.empty_cache_rate == 0:
                torch.cuda.empty_cache()
                logging.info("Cleaned CUDA cache")
            output_path = Path(path).with_suffix(args.output_ext)
            if output_path.exists():
                continue
            try:
                durations, token_ids, tokens = extractor.extract_alignment(
                    path,
                    text,
                    plot=False,
                    add_trailing_silence=True,
                )
                assert (
                    durations.shape[-1] == token_ids.shape[-1]
                ), f"Text token and duration shape mismatch: {durations.shape} != {token_ids.shape}, path={path}, text={text}"
                np.save(output_path, durations.cpu().numpy().astype(np.int64))
            except Exception as e:
                errors.append((path, text, str(e)))
                print(f"Error in {path}: {e}")  # fallback to cpu?
    if errors:
        logging.error(f"Errors: {errors}")
        with open(Path(args.input_tsv).parent / "errors.txt", "w") as f:
            for error in errors:
                f.write(f"{error}\n")
