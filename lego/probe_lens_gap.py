"""How much of what is linearly decodable does the logit lens see?

At every layer and every ``<op>``/``<predict>`` position, trains a linear probe
on the final-normed residual (what the lens reads) and compares it with the
lens's top-1 accuracy. The lens can only distinguish group elements through
the 5-dimensional *visible* subspace spanned by the six element embeddings
(minus their mean, since softmax ignores a shared offset); everything else in
the 96-dimensional residual is *dark* to it. Probes on each part separately
say where the information lives:

- lens ≈ full probe: the lens sees everything that is there;
- full probe high, visible probe low, dark probe high: computed, but dark;
- all probes at chance: not linearly present at this position at all.

A probe on a random 5-dimensional subspace is the baseline for "visible": a
visible probe no better than it means the embedding directions are not
special. Also probes for other quantities a non-sequential model might hold
at the ``<op>`` after operand j: the product of the operands alone (no start
element), the operand itself, and the product of the last two operands.

Finally, probes the ``<predict>`` position for the product of every
contiguous run of inputs (e0, g1, …, gk): a sequential model's left-to-right
prefixes t[j] versus, e.g., the pairs a tree-shaped algorithm would build.

    uv run python -m lego.probe_lens_gap \
        --checkpoint hf:lego/std_96d_6h_8L_kp2_s42_fsl_only/step_156250.pt
"""

import argparse
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from lego.analyze_logit_lens import make_logit_lens_fn, print_heatmap
from lego.generator import S3, ChainExample, compose, generate_fixed_dataset
from lego.model import AnyModel
from lego.tokenizer import element_token, encode
from lego.training import load_model

N_ELEMENTS = S3.order
CHANCE = 1 / N_ELEMENTS


def state_position(j: int) -> int:
    """Position of the ``<op>`` (or, for j = k, ``<predict>``) after operand j."""
    return 2 * (j + 1)


def _product(ops: tuple[int, ...]) -> int:
    """g_n · … · g_1 for (g_1, …, g_n): the trajectory's left-multiplication order."""
    acc = ops[0]
    for op in ops[1:]:
        acc = compose(op, acc)
    return acc


Target = Callable[[ChainExample, int], int]

TARGETS: dict[str, Target] = {
    "state t[j]": lambda ex, j: ex.trajectory[j],
    "operand product g_j…g_1": lambda ex, j: _product(ex.ops[:j]),
    "operand g_j": lambda ex, j: ex.ops[j - 1],
    "last two g_j·g_(j-1)": lambda ex, j: _product(ex.ops[max(j - 2, 0) : j]),
}


def run_product(ex: ChainExample, first: int, last: int) -> int:
    """Product of inputs first..last of (e0, g1, …, gk), in trajectory order."""
    return _product((ex.start, *ex.ops)[first : last + 1])


def run_label(first: int, last: int) -> str:
    names = ["e0"] + [f"g{i}" for i in range(1, last + 1)]
    return names[first] if first == last else f"{names[last]}…{names[first]}"


def visible_basis(model: AnyModel) -> Tensor:
    """Orthonormal (dim, 5) basis of the element embeddings' differences."""
    first = element_token(0)
    rows = model.tok_emb.weight[first : first + N_ELEMENTS].detach()
    u, _s, _vh = torch.linalg.svd((rows - rows.mean(0)).T, full_matrices=False)
    return u[:, : N_ELEMENTS - 1]


def random_basis(dim: int, rank: int, generator: torch.Generator) -> Tensor:
    q, _r = torch.linalg.qr(torch.randn(dim, rank, generator=generator))
    return q


@torch.no_grad()
def normed_residuals(model: AnyModel, examples: list[ChainExample]) -> list[Tensor]:
    """Final-norm applied to each layer's residual: (batch, seq, dim) per layer."""
    device = next(model.parameters()).device
    input_ids = torch.tensor([encode(ex) for ex in examples], device=device)
    _logits, residuals = model.forward_with_residuals(input_ids)
    return [model.final_norm(r).cpu() for r in residuals]


def probe_accuracy(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    l2: float = 1e-3,
) -> float:
    """Held-out accuracy of an L2-regularized multinomial logistic regression."""
    mean, std = x_train.mean(0), x_train.std(0).clamp_min(1e-6)
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    weight = torch.zeros(x_train.shape[1], N_ELEMENTS, requires_grad=True)
    bias = torch.zeros(N_ELEMENTS, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias], max_iter=200, line_search_fn="strong_wolfe"
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(x_train @ weight + bias, y_train)
        loss = loss + l2 * weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)  # type: ignore[arg-type]
    with torch.no_grad():
        preds = (x_test @ weight + bias).argmax(-1)
    return (preds == y_test).float().mean().item()


def _labels(examples: list[ChainExample], target: Target, j: int) -> Tensor:
    return torch.tensor([target(ex, j) for ex in examples])


def chance_corrected_ratio(lens: Tensor, probe: Tensor) -> Tensor:
    """(lens − chance) / (probe − chance); NaN where the probe is near chance."""
    excess = probe - CHANCE
    ratio = (lens - CHANCE) / excess
    return torch.where(excess > 0.1, ratio, torch.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--n-random-subspaces", type=int, default=5)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.checkpoint, device)
    k, n_layers = args.k, config.n_layers

    train = generate_fixed_dataset(k, args.n_train, seed=args.seed + 1)
    test = generate_fixed_dataset(k, args.n_test, seed=args.seed)
    res_train, res_test = normed_residuals(model, train), normed_residuals(model, test)
    lens = make_logit_lens_fn(model)

    visible = visible_basis(model).cpu()
    generator = torch.Generator().manual_seed(args.seed)
    randoms = [
        random_basis(config.dim, visible.shape[1], generator)
        for _ in range(args.n_random_subspaces)
    ]
    dark = torch.eye(config.dim) - visible @ visible.T

    def table() -> Tensor:
        return torch.zeros(n_layers, k)

    lens_acc, visible_var = table(), table()
    probes = {
        name: table() for name in ["full", "visible (5d)", "random 5d", "dark (91d)"]
    }
    alt_probes = {name: table() for name in list(TARGETS)[1:]}

    for layer in range(n_layers):
        for col, j in enumerate(range(1, k + 1)):
            pos = state_position(j)
            x_tr, x_te = res_train[layer][:, pos], res_test[layer][:, pos]
            y_tr, y_te = (_labels(d, TARGETS["state t[j]"], j) for d in (train, test))

            top1 = lens(x_te.to(device), layer).argmax(-1).cpu()
            lens_acc[layer, col] = (top1 == y_te + element_token(0)).float().mean()

            centered = x_te - x_te.mean(0)
            visible_energy = (centered @ visible).pow(2).sum()
            visible_var[layer, col] = visible_energy / centered.pow(2).sum()

            probes["full"][layer, col] = probe_accuracy(x_tr, y_tr, x_te, y_te)
            probes["visible (5d)"][layer, col] = probe_accuracy(
                x_tr @ visible, y_tr, x_te @ visible, y_te
            )
            probes["random 5d"][layer, col] = sum(
                probe_accuracy(x_tr @ q, y_tr, x_te @ q, y_te) for q in randoms
            ) / len(randoms)
            probes["dark (91d)"][layer, col] = probe_accuracy(
                x_tr @ dark, y_tr, x_te @ dark, y_te
            )
            for name in alt_probes:
                alt_probes[name][layer, col] = probe_accuracy(
                    x_tr,
                    _labels(train, TARGETS[name], j),
                    x_te,
                    _labels(test, TARGETS[name], j),
                )

    cols = [f"t[{j}]" for j in range(1, k + 1)]
    cols[-1] += "/ans"
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Chance = {CHANCE:.0%}; probes trained on {args.n_train}, tested on")
    print(f"{args.n_test} held-out examples; column t[j] = the <op> after operand j")

    print_heatmap(lens_acc, "Logit lens top-1 = t[j]", col_labels=cols)
    for name, acc in probes.items():
        print_heatmap(acc, f"Linear probe for t[j], {name} residual", col_labels=cols)
    print_heatmap(
        chance_corrected_ratio(lens_acc, probes["full"]),
        "Lens efficiency: (lens − chance) / (full probe − chance)",
        col_labels=cols,
        fmt=".2f",
    )
    print_heatmap(
        visible_var,
        f"Share of residual variance in the visible 5d subspace "
        f"(isotropic baseline {(N_ELEMENTS - 1) / config.dim:.1%})",
        col_labels=cols,
        fmt=".1%",
    )
    for name, acc in alt_probes.items():
        print_heatmap(acc, f"Linear probe (full residual) for {name}", col_labels=cols)

    predict_pos = state_position(k)
    runs = [
        (a, b)
        for length in range(2, k + 2)
        for a in range(k + 2 - length)
        for b in [a + length - 1]
    ]
    run_acc = torch.zeros(len(runs), n_layers)
    for row, (a, b) in enumerate(runs):
        y_tr = torch.tensor([run_product(ex, a, b) for ex in train])
        y_te = torch.tensor([run_product(ex, a, b) for ex in test])
        for layer in range(n_layers):
            run_acc[row, layer] = probe_accuracy(
                res_train[layer][:, predict_pos],
                y_tr,
                res_test[layer][:, predict_pos],
                y_te,
            )
    print("\nLinear probe at <predict> for each run product (rows) by layer")
    print(f"{'run':>10s}" + "".join(f"{'L' + str(li):>6s}" for li in range(n_layers)))
    for (a, b), accs in zip(runs, run_acc, strict=True):
        print(f"{run_label(a, b):>10s}" + "".join(f"{v:>6.0%}" for v in accs))


if __name__ == "__main__":
    main()
