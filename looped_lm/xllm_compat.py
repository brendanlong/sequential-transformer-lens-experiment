# xLLM and datasets live in the separate IFM env (looped_lm/jobs/ifm_env.sh).
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false
"""Run IFM's xLLM-Loop checkpoints without its compiled extensions or FlashAttention.

Needs a checkout of github.com/ifm-ai/xllm-loop on ``sys.path``. Importing this
module stubs every compiled extension xLLM imports at module level; the plain
inference path never calls them except the RMSNorm kernel, which is replaced
by a PyTorch version. ``load`` then builds a Part I artifact with PyTorch
attention (``causal_attn_backend=None``) and unfused blocks.
"""

import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path

import torch
from torch import nn

STUBBED = (
    "xllm_extension",
    "flash_attn_2_cuda",
    "fla",
    "flash_attn",
    "flash_attn_3",
    "mamba_ssm",
    "causal_conv1d",
    "grouped_gemm",
)


class _Missing:
    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attr: str) -> "_Missing":
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Missing(f"{self._name}.{attr}")

    def __call__(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError(f"stubbed native op called: {self._name}")

    def __hash__(self) -> int:
        return hash(self._name)


class _StubModule(types.ModuleType):
    def __getattr__(self, attr: str) -> _Missing:
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Missing(f"{self.__name__}.{attr}")


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(
        self, fullname: str, path: object, target: object = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname.split(".")[0] in STUBBED:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType:
        module = _StubModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module: types.ModuleType) -> None:
        pass


sys.meta_path.insert(0, _Finder())

import xllm.modules.rms_norm as _rms  # noqa: E402
from xllm.config import ModelConf, TokenizerConf  # noqa: E402
from xllm.data.dataset_streamer.tokenizer import build_tokenizer  # noqa: E402
from xllm.models.build import get_model_cls  # noqa: E402
from xllm.paper_part1.artifacts import load_artifact  # noqa: E402


def _group_rms(
    x: torch.Tensor, num_features: int, num_groups: int, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    xf = x.float().unflatten(-1, (num_groups, num_features // num_groups))
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * rstd).flatten(-2), rstd.squeeze(-1)


def _rms_fwd(
    x: torch.Tensor, num_features: int, num_groups: int, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    y, rstd = _group_rms(x, num_features, num_groups, eps)
    return y.to(x.dtype), rstd


def _rms_fwd_affine(
    x: torch.Tensor,
    num_features: int,
    num_groups: int,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    y, rstd = _group_rms(x, num_features, num_groups, eps)
    return (y * weight.float()).to(x.dtype), rstd


_rms.group_rms_norm_fwd = _rms_fwd
_rms.group_rms_norm_fwd_affine = _rms_fwd_affine


def load(
    artifact_dir: str | Path, device: str | torch.device = "cuda"
) -> tuple[nn.Module, object, ModelConf]:
    """Model (bf16, eval), tokenizer and config of a Part I artifact."""
    config, state_dict = load_artifact(artifact_dir)
    tokenizer = build_tokenizer(TokenizerConf.from_dict(config["tokenizer"]))
    cfg = ModelConf.from_dict(config["model"])
    cfg.init_mode = "none"
    cfg.fused_block = False
    cfg.fused_output_layer = False
    cfg.causal_attn_backend = None
    if cfg.vocab_size == -1:
        cfg.vocab_size = tokenizer.vocab_size
    prev_device, prev_dtype = torch.get_default_device(), torch.get_default_dtype()
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = get_model_cls(cfg.arch, cfg)(cfg, tokenizer)
    finally:
        torch.set_default_device(prev_device)
        torch.set_default_dtype(prev_dtype)
    model.load_state_dict(state_dict, strict=True)
    return model.eval(), tokenizer, cfg
