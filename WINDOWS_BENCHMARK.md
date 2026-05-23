# Benchmark chord-extractor-api `/sections` on Windows

> **Audience**: Claude Code instance running on a Windows 11 box with Docker Desktop + WSL2.
> **Goal**: Measure `POST /sections` end-to-end latency on (1) CPU native amd64 and (2) RTX 3050 Ti GPU. Compare with the Apple Silicon / Rosetta baseline of 4:32.
> **Time budget**: ~30-45 min first run (mostly Docker build + model download); ~5 min for subsequent benchmark requests.

This document is self-contained. The repo already has both `Dockerfile` (CPU) and `Dockerfile.gpu` (GPU) pre-staged, and `app/extractor.py` auto-picks `device='cuda'` when available. No code edits needed — just clone, build, run, curl.

## Test fixture

Use this YouTube URL every time so results are comparable across machines:

```
https://www.youtube.com/watch?v=JgdXcwuggpU
```

Track is 269.28 s (4:29). YouTube ID `JgdXcwuggpU`. The known-good response has `duration=269.281814`, `bpm=107`, and 10-13 segments labeled from `{intro, verse, chorus, bridge, inst, solo, break, outro}`.

## Baseline to beat

| Setup | Total `/sections` time | Notes |
|---|---|---|
| Apple Silicon Docker Desktop (Rosetta linux/amd64) | **4:32** | What was measured on Mac. Single-fold + hdemucs_mmi. |
| Native amd64 CPU (this machine, Phase 1 below) | target ~25-50s | i5 cores, no emulation. |
| RTX 3050 Ti CUDA (this machine, Phase 2 below) | target ~10-25s | GPU for both Demucs + allin1 model. |

If the numbers you measure are way off these targets, check the troubleshooting section.

## Prerequisites (one-time)

```powershell
# Verify Docker Desktop is on amd64 linux backend
docker version
docker info | Select-String -Pattern "Architecture|OSType|Total Memory"
```

Required:
- `OSType: linux`, `Architecture: x86_64`.
- Docker Desktop Memory ≥ 6 GiB (Settings → Resources → Memory). The CPU image peaks ~5 GiB during Demucs.
- Free disk ≥ 12 GiB (CPU image 3.2 GiB + GPU image ~7 GiB + scratch).

For Phase 2 (GPU) additionally:
- NVIDIA driver ≥ 535 on Windows host.
- Docker Desktop GPU support enabled (it auto-detects when driver is present; verify with the test below).

```powershell
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Expected output: a table that mentions `NVIDIA GeForce RTX 3050 Ti` and a CUDA version. If you get `could not select device driver "" with capabilities: [[gpu]]`, see troubleshooting before continuing.

## Get the code

```powershell
cd $env:USERPROFILE
git clone -b feat/meter-endpoint git@github.com:caohv/chord-extractor-api.git
cd chord-extractor-api
```

(If SSH isn't set up, swap to HTTPS: `git clone -b feat/meter-endpoint https://github.com/caohv/chord-extractor-api.git`.)

Both `Dockerfile` and `Dockerfile.gpu` are at the repo root. `WINDOWS_BENCHMARK.md` is the file you're reading now.

---

## Phase 1 — CPU native amd64

### Build

```powershell
docker buildx build --platform linux/amd64 -t chord-extractor-api:cpu .
```

Expected build time: **15-25 min** first run (PyTorch CPU wheel ~190 MB + NATTEN wheel + allin1/demucs + ~330 MB of prefetched model weights). Subsequent builds with no code change are cached and finish in <30 s.

If the build dies during the prefetch step with `Trying to use DiffQ, but diffq is not installed`, your local Dockerfile is out of date — `git pull` and rebuild.

### Run

```powershell
docker rm -f chord-test 2>$null
docker run -d --name chord-test --platform linux/amd64 -p 8000:8000 chord-extractor-api:cpu
Start-Sleep -Seconds 4
Invoke-RestMethod -Uri http://localhost:8000/health
```

Expected: `@{status=ok}`. Confirm the `/sections` route registered:

```powershell
(Invoke-RestMethod -Uri http://localhost:8000/openapi.json).paths.PSObject.Properties.Name
```

Expected output includes `/sections`. If only the 4 old routes show, the image was built from the wrong branch.

### Benchmark

```powershell
$body = '{"url": "https://www.youtube.com/watch?v=JgdXcwuggpU"}'
$start = Get-Date
$resp = Invoke-RestMethod -Uri http://localhost:8000/sections `
    -Method POST -ContentType "application/json" -Body $body -TimeoutSec 600
$elapsed = (Get-Date) - $start
"`nElapsed: $([math]::Round($elapsed.TotalSeconds, 1))s`n"
"duration: $($resp.duration)"
"bpm: $($resp.bpm)"
"beats: $($resp.beats.Count), downbeats: $($resp.downbeats.Count)"
"segments:"
$resp.segments | ForEach-Object { "  {0,7:N2} - {1,7:N2}  {2}" -f $_.start, $_.end, $_.label }
$resp | ConvertTo-Json -Depth 10 | Out-File $env:TEMP\sections-cpu.json
```

Sanity check the response: `duration` must be `269.281814`, `bpm` must be `107`. If those are off, the audio download is broken (geo-block? rate limit?) — re-run after a minute or report the error and stop.

### What to capture for the report

```powershell
"Image size: $((docker image ls chord-extractor-api:cpu --format '{{.Size}}'))"
docker stats --no-stream chord-test
```

Also grep the container logs for the Demucs and model timings:

```powershell
docker logs chord-test 2>&1 | Select-String -Pattern "Separated tracks|274.95/274.95|Extracting spectrograms.*100%|POST /sections"
```

Expected pattern from the Apple Silicon run (slower on Mac, your Windows numbers should compress these):

```
Separated tracks will be stored in /tmp/allin1-XXX/demix/hdemucs_mmi
... 100%|██████████| 274.95/274.95 [01:55<00:00, ...] # Demucs ~2 min on Mac, expect ~15-30s on i5
Extracting spectrograms: 100%|██████████| 1/1 [00:04<00:00, ...]
INFO: ... "POST /sections HTTP/1.1" 200 OK
```

### Stop

```powershell
docker rm -f chord-test
```

---

## Phase 2 — GPU CUDA on RTX 3050 Ti

### Build

```powershell
docker buildx build --platform linux/amd64 -f Dockerfile.gpu -t chord-extractor-api:gpu .
```

Expected build time: **25-40 min** first run (PyTorch CUDA wheel ~2.5 GB is the largest layer). Cached layers from the CPU build don't help here — `Dockerfile.gpu` uses a different runtime base (`nvidia/cuda:12.1.0-runtime-ubuntu22.04`) and pulls CUDA-tagged wheels.

Image size: **6-8 GB**.

### Run

```powershell
docker rm -f chord-test 2>$null
docker run -d --name chord-test --platform linux/amd64 --gpus all -p 8000:8000 chord-extractor-api:gpu
Start-Sleep -Seconds 4
Invoke-RestMethod -Uri http://localhost:8000/health
```

### Verify GPU is actually live inside the container

```powershell
docker exec chord-test /app/.pixi/envs/default/bin/python -c "import torch; print('cuda_available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('mem total:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), 'GB' if torch.cuda.is_available() else '')"
```

Expected:
```
cuda_available: True
device: NVIDIA GeForce RTX 3050 Ti
mem total: 4.0 GB
```

If `cuda_available: False`, the `--gpus all` flag didn't take effect — check Docker Desktop GPU integration.

### Benchmark

Same script as Phase 1, but save to a different file. Also kick off `nvidia-smi` in a parallel pane to capture GPU utilization during inference (optional, nice-to-have):

```powershell
$body = '{"url": "https://www.youtube.com/watch?v=JgdXcwuggpU"}'
$start = Get-Date
$resp = Invoke-RestMethod -Uri http://localhost:8000/sections `
    -Method POST -ContentType "application/json" -Body $body -TimeoutSec 300
$elapsed = (Get-Date) - $start
"`nElapsed (GPU): $([math]::Round($elapsed.TotalSeconds, 1))s`n"
"duration: $($resp.duration), bpm: $($resp.bpm)"
$resp | ConvertTo-Json -Depth 10 | Out-File $env:TEMP\sections-gpu.json
```

In another PowerShell tab during the request:

```powershell
# Watch GPU utilization; Ctrl+C when /sections returns
while ($true) { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; Start-Sleep 1 }
```

### Stop

```powershell
docker rm -f chord-test
```

---

## Reporting back

Capture both phase results in this format and reply with it:

```
== Phase 1 (CPU) ==
Host: <i5 model, clock>
Docker mem allocation: <X GB>
Build time: <Y min>
Image size: 3.2 GB
/sections elapsed: <Z>s
Demucs wall-clock: ~<a>s (from tqdm 274.95/274.95)
Spec extraction: ~<b>s
Model inference: ~<c>s (estimate from POST end minus prior stages)
Container mem peak: <d> GiB
Sanity: bpm=107 ✓, duration=269.28 ✓
First 3 segments: [label,start,end x3]

== Phase 2 (GPU) ==
GPU: NVIDIA GeForce RTX 3050 Ti (4 GB VRAM)
Driver: <version>
Build time: <Y min>
Image size: <X> GB
cuda_available inside container: True ✓
/sections elapsed: <Z>s
GPU util peak: <%>
GPU mem used peak: <X> MB
Sanity: bpm=107 ✓, duration=269.28 ✓

== Comparison vs Apple Silicon Rosetta (4:32) ==
CPU speedup: <X>×
GPU speedup: <X>×
```

Also paste the full JSON of one run (CPU is fine) so we can sanity-check the segments shape:

```powershell
Get-Content $env:TEMP\sections-cpu.json | Select-Object -First 100
```

---

## Troubleshooting

### Build fails with `no matching manifest for linux/arm64`

You're on a Windows ARM host (unusual) or the docker daemon defaulted to arm. Force amd64:

```powershell
$env:DOCKER_DEFAULT_PLATFORM = "linux/amd64"
```

Then rebuild. If the host is genuinely ARM (Surface Pro X / Snapdragon), this image won't run — NATTEN wheel is `linux_x86_64` only.

### `docker run` exits seconds after start

Container OOMed. Increase Docker Desktop memory: Settings → Resources → Memory → ≥ 6 GB. Restart Docker Desktop and re-run.

### `/sections` returns HTTP 500

`docker logs chord-test 2>&1 | Select-String -Pattern "Error|Traceback" -Context 0,10`

Common errors:
- `subprocess.CalledProcessError ... <Signals.SIGKILL: 9>`: Demucs OOMed mid-separation. Same fix as above.
- `ImportError: cannot import name 'natten1dav'`: NATTEN version mismatch with torch. The image's pip layer is out of sync — `git pull && docker build` again.
- `ImportError ... MutableSequence`: madmom compat patch didn't apply. Sanity-check that `app/extractor.py` is the one from this branch (should contain `_patch_madmom_compat` and call it inside `extract_sections`).

### yt-dlp fails with HTTP 403 or "Sign in to confirm you're not a bot"

YouTube rate-limited / IP-blocked the request. Wait 30 min and retry, or pre-download the audio file once on the host and serve it from a local URL (e.g. `python -m http.server` in another dir) to skip yt-dlp.

### Phase 2: `docker run --gpus all` fails with `could not select device driver`

NVIDIA Container Toolkit isn't wired into Docker Desktop. In Docker Desktop:
1. Settings → Resources → WSL Integration → enable for your default WSL distro.
2. In that WSL distro, ensure `nvidia-container-toolkit` is installed:
   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   ```
3. Restart Docker Desktop.

### Phase 2: `cuda_available: False` inside the container

- Confirm host `nvidia-smi` works (driver loaded).
- Confirm `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi` works (toolkit OK).
- Confirm the image was built from `Dockerfile.gpu` and not `Dockerfile`: `docker exec chord-test pip show torch | findstr "Version Location"` — version must read `2.5.0+cu121` (cpu build would say `2.5.0+cpu`).

### Phase 2: Out of GPU memory (3050 Ti has only 4 GB VRAM)

`torch.OutOfMemoryError: CUDA out of memory`. Single-fold allin1 + hdemucs_mmi typically peaks ~2.5 GB, so 4 GB should fit. But other processes can hold VRAM (browser hardware accel, Discord, games). Check from Windows host:

```powershell
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

If >1 GB is already used at idle, close apps until free memory is ≥ 3.5 GB before retrying.

---

## Reference — what's in this repo

- `app/main.py` — FastAPI routes. `/sections` calls `extract_sections` via `run_in_threadpool`.
- `app/extractor.py:extract_sections` — orchestrates: madmom + NATTEN compat patches → monkey-patch allin1's hardcoded `htdemucs` → `hdemucs_mmi` → auto-pick `device='cuda' if torch.cuda.is_available() else 'cpu'` → call `allin1.analyze()` in a per-request tempdir.
- `app/schemas.py:SectionsResponse` — `{duration, bpm, beats, downbeats, segments[start,end,label]}`.
- `Dockerfile` — CPU image. Pixi env + pip layer (torch CPU 2.5.0, NATTEN 0.17.4 CPU, allin1 1.1.0, diffq). Prefetches harmonix-all (8 folds), hdemucs_mmi, htdemucs.
- `Dockerfile.gpu` — same structure, swaps to PyTorch + NATTEN CUDA 12.1 wheels and `nvidia/cuda:12.1.0-runtime-ubuntu22.04` runtime base.
- `ALLIN1_MODEL` env var — defaults to `harmonix-fold0` (single fold). Set to `harmonix-all` for the 8-fold ensemble (~8× slower, ~1-3 F1 better).

If you need to dig deeper, the comments in `app/extractor.py` document why each patch exists (NATTEN's CUDA-probe at CPU import, allin1's `from .analyze import analyze` shadowing the module, mdx_q's 4-model-bag OOM behavior we rejected, etc).
