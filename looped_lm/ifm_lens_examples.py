# xLLM and datasets live in the separate IFM env (looped_lm/jobs/ifm_env.sh).
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false
"""Logit-lens top-k tokens at every executed block and position, for a few prompts.

One fixed factual prompt plus random FineWeb-Edu snippets, the same for every
IFM model, so the report can show the classic logit-lens grid.

    PYTHONPATH=.:../ext/xllm-loop python -m looped_lm.ifm_lens_examples \
        --artifact ../ext/ifm/dense-ouro-336b --out OUT.json
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from looped_lm import xllm_compat
from looped_lm.ifm import effective_gain, record, rms_normed

FIXED = "The Eiffel Tower is in the city of Paris, which is the capital of France"


def prompts(tokenizer: object, n_random: int, n_tokens: int, seed: int) -> list:
    from datasets import load_dataset

    out = [tokenizer.encode(FIXED, bos=True, eos=False)]
    stream = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True
    )
    rng = random.Random(seed)
    docs = []
    for doc in stream:
        ids = tokenizer.encode(doc["text"], bos=True, eos=False)
        if len(ids) > 200:
            docs.append(ids)
        if len(docs) == 200:
            break
    for ids in rng.sample(docs, n_random):
        start = rng.randrange(32, len(ids) - n_tokens)
        out.append(ids[:1] + ids[start : start + n_tokens - 1])
    return out


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-random", type=int, default=2)
    parser.add_argument("--n-tokens", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offload", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    model, tokenizer, cfg = xllm_compat.load(
        args.artifact, device="cpu" if args.offload else device
    )
    if args.offload:
        for name, child in model.named_children():
            if name != "layers":
                child.to(device)
    head = model.output
    eps = getattr(head.final_norm, "eps", cfg.rmsnorm_eps)
    gain = effective_gain(head.final_norm, cfg.model_dim, eps).to(device)
    unembed = head.output.weight.detach().float()

    def piece(i: int) -> str:
        return tokenizer.decode([i])

    result: dict = {"artifact": str(args.artifact), "prompts": []}
    for ids in prompts(tokenizer, args.n_random, args.n_tokens, args.seed):
        tokens = torch.tensor([ids])
        positions = torch.arange(len(ids))[None]
        states, layer_ids = record(model, tokens, positions, device, args.offload, 1)
        cells = []
        for state in states[:, 0]:
            z = rms_normed(state.to(device), eps)
            probs = F.softmax((z * gain) @ unembed.T, dim=-1)
            top = probs.topk(args.top_k, dim=-1)
            cells.append(
                [
                    [
                        [piece(i), round(p, 4)]
                        for i, p in zip(
                            top.indices[pos].tolist(),
                            top.values[pos].tolist(),
                            strict=True,
                        )
                    ]
                    for pos in range(len(ids))
                ]
            )
        result["prompts"].append(
            {
                "tokens": [piece(i) for i in ids],
                "block_ids": layer_ids,
                "layers": cells,  # [execution][position] -> top-k [token, prob]
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
