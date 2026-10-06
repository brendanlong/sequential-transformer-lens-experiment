"""Logit-lens top-k tokens at every layer and position for a few random prompts.

The same held-out compositions (test ID) for every model, so the report can show
how the lens looks on concrete prompts.

    uv run python -m looped_lm.twohop_lens_examples --root ext/kohli \
        --out results/looped-lm/twohop-lens-examples.json
"""

import argparse
import json
from pathlib import Path

import torch

from looped_lm.twohop import load_data, load_model

MODELS = [
    "checkpoints/systematicity/r1_l8/checkpoint_epoch_2001.pt",
    "checkpoints/systematicity/r1_l4/checkpoint_epoch_13501.pt",
    "checkpoints/systematicity/r2_l4/checkpoint_epoch_1201.pt",
    "checkpoints/systematicity/r2_l4/checkpoint_epoch_7101.pt",
    "checkpoints/systematicity/r4_l4/checkpoint_epoch_401.pt",
    "checkpoints/systematicity/r4_l4/checkpoint_epoch_3501.pt",
    "checkpoints/systematicity/r8_l4/checkpoint_epoch_301.pt",
    "checkpoints/systematicity/r8_l4/checkpoint_epoch_2901.pt",
]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--root", default="ext/kohli")
    parser.add_argument("--n-examples", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = Path(args.root)
    vocab, _probe, evals = load_data(root, 1, args.n_examples, args.seed + 1)
    names = {i: t for t, i in vocab.items()}
    ex = evals["test_inferred_iid"]
    prompts = [
        {"tokens": [names[i] for i in ids], "bridge": names[b], "answer": names[t]}
        for ids, b, t in zip(
            ex.ids.tolist(), ex.bridge.tolist(), ex.tail.tolist(), strict=True
        )
    ]
    out: dict = {"prompts": prompts, "models": {}}
    for ckpt in MODELS:
        model, epoch = load_model(root / ckpt, torch.device("cpu"))
        residuals = model.residuals(ex.ids)
        layers = []
        for x in residuals:
            probs = model.lens(x).softmax(-1)  # (n, 3, vocab)
            top = probs.topk(args.top_k, dim=-1)
            bridge_p = probs.gather(-1, ex.bridge[:, None, None].expand(-1, 3, 1))
            answer_p = probs.gather(-1, ex.tail[:, None, None].expand(-1, 3, 1))
            layers.append(
                [
                    [
                        {
                            "top": [
                                [names[i], round(p, 4)]
                                for i, p in zip(
                                    top.indices[e, pos].tolist(),
                                    top.values[e, pos].tolist(),
                                    strict=True,
                                )
                            ],
                            "p_bridge": round(bridge_p[e, pos, 0].item(), 4),
                            "p_answer": round(answer_p[e, pos, 0].item(), 4),
                        }
                        for pos in range(3)
                    ]
                    for e in range(len(prompts))
                ]
            )
        final = model.lens(residuals[-1][:, 2]).argmax(-1)
        out["models"][ckpt] = {
            "epoch": epoch,
            "iterations": model.n_iterations,
            "block_layers": len(model.blocks),
            "correct": (final == ex.tail).tolist(),
            "layers": layers,  # [layer][example][position]
        }
        print(ckpt, "correct", sum(out["models"][ckpt]["correct"]), "/", len(prompts))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
