# xLLM and datasets live in the separate IFM env (looped_lm/jobs/ifm_env.sh).
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false
"""How much of the "dark subspace" is a few massive residual coordinates?

At every executed block, on FineWeb-Edu positions: the share of the normed
residual's energy in each of its largest coordinates ("sinks"), and the share
in the most weakly read quarter of readout directions with and without the
sink coordinates (any coordinate holding more than ``--sink-share`` of the
energy at that block). Also relative visibility with the sinks removed.

    PYTHONPATH=.:../ext/xllm-loop python -m looped_lm.ifm_sink \
        --artifact ../ext/ifm/dense-ouro-336b --out OUT.json
"""

import argparse
import gc
import json
import random
from pathlib import Path

import torch

from looped_lm import lens_metrics as lm
from looped_lm import xllm_compat
from looped_lm.ifm import effective_gain, load_text_tokens, record, rms_normed


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-seq", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--n-pos", type=int, default=64)
    parser.add_argument("--sink-share", type=float, default=0.05)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda")
    model, tokenizer, cfg = xllm_compat.load(
        args.artifact, device="cpu" if args.offload else device
    )
    if args.offload:
        for name, child in model.named_children():
            if name != "layers":
                child.to(device)
    tokens = load_text_tokens(tokenizer, args.n_seq, args.seq_len)
    rng = random.Random(args.seed)
    positions = torch.tensor(
        [
            sorted(rng.sample(range(16, args.seq_len - 1), args.n_pos))
            for _ in range(args.n_seq)
        ]
    )
    states, layer_ids = record(model, tokens, positions, device, args.offload, 2)
    eps = getattr(model.output.final_norm, "eps", cfg.rmsnorm_eps)
    gain = effective_gain(model.output.final_norm, cfg.model_dim, eps).cpu()
    unembed = model.output.output.weight.detach().to("cpu", torch.float32)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    s, vh = lm.singular_basis(lm.folded_readout(unembed, gain, centers=False))
    s, vh = s.to(device), vh.to(device)
    bottom = vh[3 * len(vh) // 4 :]

    rows = []
    for e in range(states.shape[0]):
        z = rms_normed(states[e].flatten(0, 1).to(device), eps)
        energy = z.pow(2).mean(0)
        share = energy / energy.sum()
        order = share.argsort(descending=True)
        sinks = [int(i) for i in order if share[i] > args.sink_share]
        z_ns = z.clone()
        z_ns[:, sinks] = 0
        rows.append(
            {
                "execution": e,
                "block": layer_ids[e],
                "top_coords": [[int(i), share[i].item()] for i in order[:5]],
                "sinks": sinks,
                "sink_share": share[sinks].sum().item() if sinks else 0.0,
                "dark_raw": ((z @ bottom.T).pow(2).sum() / z.pow(2).sum()).item(),
                "dark_no_sink": (
                    (z_ns @ bottom.T).pow(2).sum() / z_ns.pow(2).sum()
                ).item(),
                "visibility_no_sink_centered": lm.relative_visibility(
                    z_ns - z_ns.mean(0), s, vh
                ),
            }
        )
        r = rows[-1]
        print(
            f"{e:>3} block {layer_ids[e]:>3} sinks {sinks} share {r['sink_share']:.2f} "
            f"dark {r['dark_raw']:.2f} -> {r['dark_no_sink']:.2f}",
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"artifact": str(args.artifact), "rows": rows}))


if __name__ == "__main__":
    main()
