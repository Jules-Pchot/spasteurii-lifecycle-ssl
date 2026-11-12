"""DINOv2 (with registers) weights from Hugging Face, converted to the torch.hub format that PhenoMe loads.

Why: PhenoMe calls torch.hub.load("facebookresearch/dinov2", ...), which downloads the weights from
dl.fbaipublicfiles.com. That server's SSL certificate expired on 2026-11-11, so the download fails. Meta also
publishes the same weights on Hugging Face (facebook/dinov2-with-registers-*), in the transformers layout.

What it does:
  1. downloads the official Hugging Face model;
  2. renames its parameters to the torch.hub layout (q / k / v projections concatenated into one qkv);
  3. checks that the converted torch.hub model gives the same output as the Hugging Face model on random images
     (max abs difference of the CLS embedding < TOLERANCE), otherwise writes nothing;
  4. saves the file in the torch hub cache under the name torch.hub looks for, so PhenoMe loads it without
     downloading anything. An existing file is never overwritten.

Usage (PhenoMe environment, from the project root):
  .venv-phenome/bin/python Video_Process_Script/Shape_Simulator/dinov2_from_hf.py small    # or base, large
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModel

# size -> (Hugging Face repo, torch.hub model name, torch.hub checkpoint file)
MODELS = {
    "small": ("facebook/dinov2-with-registers-small", "dinov2_vits14_reg", "dinov2_vits14_reg4_pretrain.pth"),
    "base": ("facebook/dinov2-with-registers-base", "dinov2_vitb14_reg", "dinov2_vitb14_reg4_pretrain.pth"),
    "large": ("facebook/dinov2-with-registers-large", "dinov2_vitl14_reg", "dinov2_vitl14_reg4_pretrain.pth"),
}
# giant is left out: its MLP (SwiGLU) has another layout in the two formats
TOLERANCE = 1e-4
N_TEST_IMAGES = 2


def convert(hf_sd):
    """transformers state dict -> torch.hub DINOv2 state dict."""
    out = {
        "cls_token": hf_sd["embeddings.cls_token"],
        "mask_token": hf_sd["embeddings.mask_token"],
        "pos_embed": hf_sd["embeddings.position_embeddings"],
        "register_tokens": hf_sd["embeddings.register_tokens"],
        "patch_embed.proj.weight": hf_sd["embeddings.patch_embeddings.projection.weight"],
        "patch_embed.proj.bias": hf_sd["embeddings.patch_embeddings.projection.bias"],
        "norm.weight": hf_sd["layernorm.weight"],
        "norm.bias": hf_sd["layernorm.bias"],
    }
    n_layers = 1 + max(int(k.split(".")[2]) for k in hf_sd if k.startswith("encoder.layer."))
    for i in range(n_layers):
        h, b = f"encoder.layer.{i}.", f"blocks.{i}."
        att = h + "attention."
        for t in ("weight", "bias"):
            out[b + f"norm1.{t}"] = hf_sd[h + f"norm1.{t}"]
            out[b + f"norm2.{t}"] = hf_sd[h + f"norm2.{t}"]
            out[b + f"attn.qkv.{t}"] = torch.cat([hf_sd[att + f"{p}_proj.{t}"] for p in "qkv"], dim=0)
            out[b + f"attn.proj.{t}"] = hf_sd[att + f"o_proj.{t}"]
            out[b + f"mlp.fc1.{t}"] = hf_sd[h + f"mlp.fc1.{t}"]
            out[b + f"mlp.fc2.{t}"] = hf_sd[h + f"mlp.fc2.{t}"]
        out[b + "ls1.gamma"] = hf_sd[h + "layer_scale1.lambda1"]
        out[b + "ls2.gamma"] = hf_sd[h + "layer_scale2.lambda1"]
    return out


def main(size):
    repo, hub_name, filename = MODELS[size]
    target = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if target.exists():
        print(f"{target} already exists: nothing to do")
        return

    hf = AutoModel.from_pretrained(repo).eval()
    sd = convert(hf.state_dict())
    hub = torch.hub.load("facebookresearch/dinov2", hub_name, pretrained=False).eval()
    hub.load_state_dict(sd, strict=True)

    x = torch.rand(N_TEST_IMAGES, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = hf(pixel_values=x).pooler_output  # layer-normed CLS token
        got = hub(x)                            # torch.hub forward = layer-normed CLS token
    diff = (ref - got).abs().max().item()
    print(f"{size}: max |CLS difference| HF vs converted = {diff:.2e} (tolerance {TOLERANCE:g})")
    if diff > TOLERANCE:
        raise SystemExit("conversion check failed: nothing written")

    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, target)
    print(f"-> {target}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("size", choices=list(MODELS), nargs="?", default="small")
    main(ap.parse_args().size)
