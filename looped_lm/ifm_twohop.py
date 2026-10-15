# xLLM and datasets live in the separate IFM env (looped_lm/jobs/ifm_env.sh).
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false
"""Is the bridge entity visible in IFM's models on natural two-hop questions?

Prompts from TwoHopFact (Yang et al. 2024, CC-BY-4.0), e.g. "The author of the
novel Nineteen Eighty-Four was born in the city of" (bridge "George Orwell",
answer "Motihari"). At every executed block, at two positions:

- **desc**: the last token of the descriptive mention ("…Eighty-Four"), where
  Yang et al. find the bridge is recalled;
- **last**: the final token, where the answer is predicted;

record the rank of the bridge's and the answer's first token, and of a control
token (another prompt's bridge of the same category), under

- the **logit lens** (final norm + unembedding), and
- an **embedding lens**: cosine between the normed residual and each token's
  *input* embedding (mean-centered over the vocabulary). IFM's models do not
  tie input and output embeddings, so a state written in input format would be
  invisible to the logit lens.

Also whether the model answers the one-hop question (does it know the bridge?)
and the two-hop question correctly, so results can be split by those.

    PYTHONPATH=.:../ext/xllm-loop python -m looped_lm.ifm_twohop \
        --artifact ../ext/ifm/dense-ouro-336b --out OUT.json
"""

import argparse
import gc
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from looped_lm import xllm_compat
from looped_lm.ifm import effective_gain, record, rms_normed

DESC = "r2(r1(e1)).subject_cut.prompt"
TWO_HOP = "r2(r1(e1)).prompt"
ONE_HOP = "r1(e1).prompt"


def first_token(tokenizer: object, prompt: str, continuation: str) -> int:
    """The first token of `continuation` as the model would emit it after `prompt`."""
    base = tokenizer.encode(prompt, bos=True, eos=False)
    full = tokenizer.encode(prompt + " " + continuation, bos=True, eos=False)
    assert full[: len(base)] == base, prompt
    return full[len(base)]


def load_items(tokenizer: object, n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    rows = load_dataset("soheeyang/TwoHopFact", split="train")
    rng = random.Random(seed)
    items = []
    for i in rng.sample(range(len(rows)), n):
        r = rows[i]
        ids = tokenizer.encode(r[TWO_HOP], bos=True, eos=False)
        desc = tokenizer.encode(r[DESC], bos=True, eos=False)
        if ids[: len(desc)] != desc:
            continue
        items.append(
            {
                "uid": r["uid"],
                "prompt": r[TWO_HOP],
                "bridge": r["e2.value"],
                "answer": r["e3.value"],
                "ids": ids,
                "one_hop_ids": tokenizer.encode(r[ONE_HOP], bos=True, eos=False),
                "desc_pos": len(desc) - 1,
                "bridge_tok": first_token(tokenizer, r[ONE_HOP], r["e2.value"]),
                "answer_tok": first_token(tokenizer, r[TWO_HOP], r["e3.value"]),
                "bridge_category": r["e2.category"],
            }
        )
    # Control: the bridge of another prompt with the same bridge category (another
    # author, another country, ...) and a different first token. Token frequency
    # and type affect it like the real bridge, but it has nothing to do with this
    # prompt.
    by_cat: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        by_cat.setdefault(it["bridge_category"], []).append(i)
    for it in items:
        pool = [
            items[j]["bridge_tok"]
            for j in by_cat[it["bridge_category"]]
            if items[j]["bridge_tok"] != it["bridge_tok"]
        ]
        it["control_tok"] = rng.choice(pool) if pool else -1
    return [it for it in items if it["control_tok"] >= 0]


def pad(seqs: list[list[int]]) -> Tensor:
    """Right-padded batch; causal attention keeps padding out of real positions."""
    width = max(len(s) for s in seqs)
    return torch.tensor([s + [0] * (width - len(s)) for s in seqs])


def ranks(score_fn: object, x: Tensor, targets: Tensor, chunk: int = 128) -> list[int]:
    """1-based rank of each row's target token among score_fn(x)'s columns."""
    out = []
    for s in range(0, len(x), chunk):
        scores = score_fn(x[s : s + chunk])  # type: ignore[operator]
        own = scores.gather(-1, targets[s : s + chunk, None])
        out += ((scores > own).sum(-1) + 1).tolist()
    return out


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=16)
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
    unembed_cpu = head.output.weight.detach().to("cpu", torch.float32)
    embed_cpu = model.embed.weight.detach().to("cpu", torch.float32)

    items = load_items(tokenizer, args.n, args.seed)
    print(f"{len(items)} prompts", flush=True)

    def states_at(
        model: torch.nn.Module, seqs: list[list[int]], pos: list[list[int]]
    ) -> tuple[Tensor, list[int]]:
        chunks, layer_ids = [], []
        for s in range(0, len(seqs), args.batch):
            st, layer_ids = record(
                model,
                pad(seqs[s : s + args.batch]),
                torch.tensor(pos[s : s + args.batch]),
                device,
                args.offload,
                args.batch,
            )
            chunks.append(st)
        return torch.cat(chunks, dim=1), layer_ids  # (executions+1, n, P, dim)

    two, layer_ids = states_at(
        model,
        [it["ids"] for it in items],
        [[it["desc_pos"], len(it["ids"]) - 1] for it in items],
    )
    one, _ = states_at(
        model,
        [it["one_hop_ids"] for it in items],
        [[len(it["one_hop_ids"]) - 1] for it in items],
    )
    del model, head
    gc.collect()  # the recording hooks leave reference cycles
    torch.cuda.empty_cache()
    unembed = unembed_cpu.to(device)
    embed = embed_cpu.to(device)
    embed_hat = F.normalize(embed - embed.mean(0), dim=-1)
    del embed, embed_cpu, unembed_cpu
    bridge = torch.tensor([it["bridge_tok"] for it in items], device=device)
    answer = torch.tensor([it["answer_tok"] for it in items], device=device)
    control = torch.tensor([it["control_tok"] for it in items], device=device)

    def logits(x: Tensor) -> Tensor:
        return (rms_normed(x.to(device), eps) * gain) @ unembed.T

    def embed_scores(x: Tensor) -> Tensor:
        return rms_normed(x.to(device), eps) @ embed_hat.T

    one_hop_rank = ranks(logits, one[-1, :, 0], bridge)
    two_hop_rank = ranks(logits, two[-1, :, 1], answer)
    layers = []
    for e in range(two.shape[0]):
        row: dict = {"execution": e, "block": layer_ids[e]}
        for p_name, p in (("desc", 0), ("last", 1)):
            for lens, fn in (("logit", logits), ("embed", embed_scores)):
                x = two[e, :, p]
                row[f"{p_name}_{lens}_bridge_rank"] = ranks(fn, x, bridge)
                row[f"{p_name}_{lens}_answer_rank"] = ranks(fn, x, answer)
                row[f"{p_name}_{lens}_control_rank"] = ranks(fn, x, control)
        layers.append(row)
        print(f"{e:>3} block {layer_ids[e]:>3}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "artifact": str(args.artifact),
                "items": [
                    {k: v for k, v in it.items() if k not in ("ids", "one_hop_ids")}
                    for it in items
                ],
                "one_hop_rank": one_hop_rank,  # bridge's rank after the one-hop prompt
                "two_hop_rank": two_hop_rank,  # answer's rank after the two-hop prompt
                "layers": layers,
            }
        )
    )


if __name__ == "__main__":
    main()
