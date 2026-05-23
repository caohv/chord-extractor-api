FROM ghcr.io/prefix-dev/pixi:latest AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pixi.toml pixi.lock* ./
RUN pixi install --locked || pixi install
RUN pixi shell-hook -e default > /shell-hook && echo 'exec "$@"' >> /shell-hook

# Layer allin1 on top of the pixi env via pip. We can't pixi-manage these:
# - PyTorch CPU wheels live on a separate index, not PyPI proper.
# - NATTEN ships CPU wheels only via SHI-Labs's GitHub releases, and allin1
#   1.1.0 only auto-requires NATTEN on macOS (not Linux), so we install a
#   torch-version-matched CPU wheel ourselves. allin1's `dinat.py` imports
#   `natten.functional.natten1dav` / `natten1dqkrpb` (and 2d variants) —
#   legacy names that NATTEN 0.17.4 still keeps as deprecation shims but
#   0.17.5 removed. Pin 0.17.4 to keep that API; PyTorch must then be
#   2.5.0 to match the NATTEN wheel's `+torch250cpu` tag.
# - Order matters: torch first (so NATTEN's wheel finds it), then NATTEN
#   (validates against installed torch), then allin1 (pulls demucs
#   transitively against the already-pinned torch).
RUN /app/.pixi/envs/default/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch==2.5.0 torchaudio==2.5.0 \
 && /app/.pixi/envs/default/bin/pip install --no-cache-dir \
        https://github.com/SHI-Labs/NATTEN/releases/download/v0.17.4/natten-0.17.4%2Btorch250cpu-cp311-cp311-linux_x86_64.whl \
 && /app/.pixi/envs/default/bin/pip install --no-cache-dir allin1==1.1.0 diffq

# Pre-fetch model weights so the first /sections request doesn't pay the
# download cost. allin1's harmonix-all caches all 8 fold checkpoints
# (~80 MB), which also covers any single `harmonix-foldN` selection.
# Demucs prefetch covers both hdemucs_mmi (the runtime default, ~80 MB)
# and htdemucs (~250 MB) so either separator works without a cold-start
# download. mdx-family models were tried but their 4-model bag layout
# peaks RAM at ~6 GB, OOM-ing on Cloudflare Containers standard-1 (4 GB).
#
# allin1 transitively imports madmom (via its spectrogram module), which
# breaks on Python 3.11 without the same compat shim used at runtime for
# /meter. Reuse app.extractor._patch_madmom_compat by copying app/ in
# first.
COPY app/ ./app/
ENV HF_HOME=/opt/cache/huggingface
ENV TORCH_HOME=/opt/cache/torch
RUN /app/.pixi/envs/default/bin/python -c "\
from app.extractor import _patch_madmom_compat, _patch_natten_cpu; \
_patch_madmom_compat(); \
_patch_natten_cpu(); \
from allin1.models import load_pretrained_model; \
load_pretrained_model('harmonix-all', device='cpu'); \
from demucs.pretrained import get_model; \
get_model('hdemucs_mmi'); \
get_model('htdemucs')"

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /app/.pixi /app/.pixi
COPY --from=build /opt/cache /opt/cache
COPY --from=build /shell-hook /shell-hook
COPY app/ ./app/
COPY pixi.toml ./

ENV PATH="/app/.pixi/envs/default/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/opt/cache/huggingface
ENV TORCH_HOME=/opt/cache/torch

EXPOSE 8000

ENTRYPOINT ["/bin/bash", "/shell-hook"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
