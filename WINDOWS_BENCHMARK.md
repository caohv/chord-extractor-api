# Benchmark chord-extractor-api /sections endpoint on Windows

## Mục tiêu

Đo thời gian thực tế của `POST /sections` trên máy Windows host (Intel i5 + RTX 3050 Ti) cho 1 audio YouTube ~4:29. Repo hiện tại đã verified chạy đúng dưới Docker Desktop trên Apple Silicon (Rosetta emulation: 4:32/request). Mục tiêu là đo:

1. **CPU native amd64** trên Windows Docker Desktop / WSL2 (kỳ vọng ~25-50s).
2. **GPU CUDA** dùng RTX 3050 Ti (kỳ vọng ~10-15s, cần Dockerfile variant mới).

## Stack đang dùng

- Python 3.11 + FastAPI + uvicorn
- `allin1==1.1.0` (music structure analyzer, ISMIR 2023) — single-fold model `harmonix-fold0` mặc định (8× nhanh hơn ensemble)
- Demucs source separation — monkey-patched dùng `hdemucs_mmi` (nhanh hơn htdemucs 2-3×, RAM ~1 GB, không OOM trên 4 GB hosts)
- PyTorch CPU 2.5.0 + torchaudio 2.5.0 + NATTEN 0.17.4 CPU wheel
- Pixi (deps), Docker (deploy)

Endpoint khác (`/extract`, `/bpm`, `/meter`) cũng tồn tại — không cần test cho mục tiêu này.

## Test input — luôn dùng URL này để so sánh

```
https://www.youtube.com/watch?v=JgdXcwuggpU
```

Track dài 269.28s (4:29). YouTube ID `JgdXcwuggpU`.

## Expected response schema

```json
{
  "duration": 269.281814,
  "bpm": 107,
  "beats": [/* ~370-400 floats */],
  "downbeats": [/* ~95-105 floats */],
  "segments": [
    { "start": 19.49, "end": 37.87, "label": "verse" },
    { "start": 37.87, "end": 55.65, "label": "intro" },
    /* ... 10-13 segments, labels in: intro, verse, chorus, bridge, inst, solo, break, outro */
  ]
}
```

`bpm` luôn 107, `duration` luôn 269.28. Segments có thể khác nhẹ giữa các runs (model fold0 deterministic nhưng có vài boundary phụ thuộc input precision).

---

## Phase 1: CPU native amd64 trên Windows

### Yêu cầu

- Windows 11 (hoặc Windows 10 với WSL2)
- Docker Desktop ≥ 4.30
- WSL2 backend enabled trong Docker Desktop settings
- Memory cấp cho Docker Desktop ≥ 6 GB (vì image runtime peak ~5 GB)
- Free disk ≥ 8 GB (image 3.2 GB + scratch space)

Verify:

```powershell
docker version
docker info | Select-String -Pattern "Architecture|OSType|Total Memory"
```

Output cần: `OSType: linux`, `Architecture: x86_64`. Total Memory ≥ 6 GiB.

### Get the code

Repo path local: `~/Workspace/weebuild/chord-extractor-api` trên Mac của user. Có 2 cách bring code sang Windows:

**Cách A — clone từ git remote (nếu user đã push)**:

```powershell
cd $env:USERPROFILE
git clone <repo-url> chord-extractor-api
cd chord-extractor-api
```

Branch cần là branch hiện tại của user (chứa monkey-patch hdemucs_mmi và env var ALLIN1_MODEL). Hỏi user nếu chưa rõ branch.

**Cách B — copy qua mạng**:

User zip thư mục từ Mac:

```bash
# Trên Mac
cd ~/Workspace/weebuild
zip -r chord-extractor-api.zip chord-extractor-api -x 'chord-extractor-api/.pixi/*' 'chord-extractor-api/.git/*'
```

Transfer zip sang Windows (USB / iCloud / scp / cloud sync). Trên Windows:

```powershell
Expand-Archive .\chord-extractor-api.zip $env:USERPROFILE\
cd $env:USERPROFILE\chord-extractor-api
```

### Build image (CPU)

```powershell
cd $env:USERPROFILE\chord-extractor-api
docker buildx build --platform linux/amd64 -t chord-extractor-api:cpu .
```

Build time ước **15-25 phút lần đầu** (download PyTorch CPU 2.5.0 ~190 MB + NATTEN wheel + allin1 + Demucs + transitives ~600 MB, prefetch model weights ~330 MB). Lần sau cached lại 1-2 phút.

Lưu ý: `--platform linux/amd64` quan trọng vì Dockerfile pin NATTEN wheel cho `linux_x86_64`. Nếu thiếu flag và Docker chọn arm64 thì sẽ fail.

### Run container

```powershell
docker rm -f chord-test 2>$null
docker run -d --name chord-test --platform linux/amd64 -p 8000:8000 chord-extractor-api:cpu
Start-Sleep -Seconds 3
Invoke-RestMethod -Uri http://localhost:8000/health
```

Expected: `@{status=ok}`.

Verify route registered:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/openapi.json | ConvertTo-Json -Depth 5 | Select-String "/sections"
```

### Run /sections benchmark

```powershell
$body = '{"url": "https://www.youtube.com/watch?v=JgdXcwuggpU"}'
$start = Get-Date
$resp = Invoke-RestMethod -Uri http://localhost:8000/sections `
    -Method POST `
    -ContentType "application/json" `
    -Body $body `
    -TimeoutSec 600
$elapsed = (Get-Date) - $start
Write-Host "Elapsed: $($elapsed.TotalSeconds)s"
$resp | ConvertTo-Json -Depth 10 | Out-File C:\temp\sections-cpu.json
Write-Host "duration: $($resp.duration), bpm: $($resp.bpm)"
Write-Host "beats: $($resp.beats.Count), downbeats: $($resp.downbeats.Count)"
Write-Host "segments:"
$resp.segments | ForEach-Object { "  {0,7:N2} - {1,7:N2}  {2}" -f $_.start, $_.end, $_.label }
```

### Expected timing (CPU)

- i5 cores hiện đại (8 cores, 3+ GHz, AVX2): **~25-50s** total
- Trên các CPU yếu hơn (i5 thế hệ 8/9, 4 cores): có thể 60-90s

Log break-down (bằng `docker logs chord-test`):

- `Separated tracks will be stored in /tmp/allin1-XXX/demix/hdemucs_mmi` → Demucs hdemucs_mmi đã được patch đúng
- `100%|██████████| 274.95/274.95 [00:XX<00:00, ...seconds/s]` → Demucs done; mong đợi ~15-30s
- `Extracting spectrograms: 100%|██████████| 1/1 [00:0X<00:00, ...]` → spec extraction ~1-2s
- Sau đó ~132 NATTEN deprecation warnings (single-fold đúng) → model inference ~10-25s
- Cuối cùng `INFO: ... POST /sections HTTP/1.1 200 OK`

### Stop container

```powershell
docker rm -f chord-test
```

---

## Phase 2: GPU CUDA trên RTX 3050 Ti

### Yêu cầu thêm so với Phase 1

- NVIDIA driver ≥ 535 trên Windows host
- WSL2 với NVIDIA CUDA support (Windows 11 hoặc Win10 21H2+ đã có sẵn)
- NVIDIA Container Toolkit cài qua Docker Desktop hoặc manually trong WSL2

Verify GPU accessible từ Docker:

```powershell
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Output cần show GPU `NVIDIA GeForce RTX 3050 Ti` với CUDA Version.

Nếu fail: setup [Docker Desktop GPU support](https://docs.docker.com/desktop/features/gpu/) trước.

### Tạo Dockerfile.gpu

Trong thư mục repo, copy `Dockerfile` thành `Dockerfile.gpu` và apply các thay đổi sau:

**Diff cần áp dụng**:

```diff
@@ stage 2: runtime base @@
-FROM debian:bookworm-slim
+FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
 RUN apt-get update && apt-get install -y --no-install-recommends \
     libsndfile1 ffmpeg \
     && rm -rf /var/lib/apt/lists/*

@@ stage 1: pip install layer @@
 RUN /app/.pixi/envs/default/bin/pip install --no-cache-dir \
-        --index-url https://download.pytorch.org/whl/cpu \
+        --index-url https://download.pytorch.org/whl/cu121 \
         --extra-index-url https://pypi.org/simple \
         torch==2.5.0 torchaudio==2.5.0 \
  && /app/.pixi/envs/default/bin/pip install --no-cache-dir \
-        https://github.com/SHI-Labs/NATTEN/releases/download/v0.17.4/natten-0.17.4%2Btorch250cpu-cp311-cp311-linux_x86_64.whl \
+        https://github.com/SHI-Labs/NATTEN/releases/download/v0.17.4/natten-0.17.4%2Btorch250cu121-cp311-cp311-linux_x86_64.whl \
  && /app/.pixi/envs/default/bin/pip install --no-cache-dir allin1==1.1.0 diffq
```

(2 thay đổi nhỏ: index URL cpu→cu121, NATTEN wheel cpu→cu121. Base image debian→nvidia/cuda.)

Lưu ý: KHÔNG thay đổi prefetch step — model load với device='cpu' vẫn OK ở build time, runtime sẽ dùng cuda.

### Sửa code để dùng CUDA runtime

File `app/extractor.py`, function `extract_sections`, hiện tại có:

```python
result = allin1.analyze(
    audio_path,
    model=model_name,
    demix_dir=scratch_path / "demix",
    spec_dir=scratch_path / "spec",
    device="cpu",   # ← hardcoded
    keep_byproducts=False,
    multiprocess=False,
)
```

Thay `device="cpu"` thành:

```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Và:

```python
result = allin1.analyze(
    audio_path,
    model=model_name,
    demix_dir=scratch_path / "demix",
    spec_dir=scratch_path / "spec",
    device=device,
    keep_byproducts=False,
    multiprocess=False,
)
```

Trong monkey-patch `_demix` (cùng file), `--device` arg cũng cần dùng `device` được truyền vào — code hiện tại đã làm đúng vì nó dùng `str(device)`.

### Bỏ NATTEN CUDA-probe stub khi có CUDA

Function `_patch_natten_cpu()` chỉ stub `torch.cuda.get_device_capability` nếu `torch.cuda.is_available()` == False. Trên GPU host, `is_available()` returns True → stub không apply → NATTEN dùng path thật. OK không cần đổi.

### Build GPU image

```powershell
cd $env:USERPROFILE\chord-extractor-api
docker buildx build --platform linux/amd64 -f Dockerfile.gpu -t chord-extractor-api:gpu .
```

Build time **~25-40 phút** lần đầu. PyTorch CUDA wheel ~2.5 GB là khoản nặng nhất.

Image size dự kiến: **6-8 GB**.

### Run với --gpus all

```powershell
docker rm -f chord-test 2>$null
docker run -d --name chord-test --platform linux/amd64 --gpus all -p 8000:8000 chord-extractor-api:gpu
Start-Sleep -Seconds 3
Invoke-RestMethod -Uri http://localhost:8000/health
```

Verify GPU detected từ inside container:

```powershell
docker exec chord-test /app/.pixi/envs/default/bin/python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expected: `cuda available: True`, device `NVIDIA GeForce RTX 3050 Ti`.

### Run /sections benchmark GPU

Same command as CPU phase. Save kết quả vào file khác:

```powershell
$body = '{"url": "https://www.youtube.com/watch?v=JgdXcwuggpU"}'
$start = Get-Date
$resp = Invoke-RestMethod -Uri http://localhost:8000/sections `
    -Method POST -ContentType "application/json" -Body $body -TimeoutSec 300
$elapsed = (Get-Date) - $start
Write-Host "Elapsed (GPU): $($elapsed.TotalSeconds)s"
$resp | ConvertTo-Json -Depth 10 | Out-File C:\temp\sections-gpu.json
```

### Expected timing (GPU 3050 Ti)

- Demucs hdemucs_mmi CUDA: **~3-7s**
- Model single-fold CUDA: **~3-7s**
- yt-dl + spec + response: ~5-10s
- **Tổng: ~10-25s** (dao động phụ thuộc GPU load, network cho yt-dl)

Log breakdown:

- `Separated tracks will be stored in /tmp/.../hdemucs_mmi` ✓
- `Selected model is a bag of 1 models` ✓
- Demucs progress bar ~5s thay vì 30s ✓
- KHÔNG còn `WARNING:natten.functional:You're calling NATTEN op natten1dav, which is deprecated` (CUDA path không qua deprecation shim trong NATTEN 0.17.4) ← hoặc vẫn có nhưng nhanh
- Model inference dưới 10s

---

## Báo cáo kết quả

Sau khi chạy xong cả 2 phases, output cần collect:

1. **Phase 1 (CPU)**:
   - `Elapsed: XXs`
   - Image size: `docker image ls chord-extractor-api:cpu --format '{{.Size}}'`
   - Container memory peak: chạy `docker stats --no-stream chord-test` giữa lúc inference
   - First 5 segments + label distribution

2. **Phase 2 (GPU)**:
   - `Elapsed: XXs`
   - Image size: `docker image ls chord-extractor-api:gpu --format '{{.Size}}'`
   - GPU utilization: chạy `nvidia-smi` giữa lúc inference từ Windows host
   - Verify `torch.cuda.is_available()` returned True
   - Confirm BPM/duration giống CPU run (sanity check)

Format report:

```
CPU run:
  Elapsed: 33.4s
  Image: 3.2 GB
  Peak mem: 1.8 GiB
  Sample segments: [intro 0-19, verse 19-37, ...]

GPU run:
  Elapsed: 12.1s
  Image: 7.1 GB
  GPU util peak: 78%
  CUDA detected: True
  Same BPM/duration: ✓
```

---

## Troubleshooting

### CPU phase

**Build fails với "no matching manifest for linux/arm64"**

Bạn quên `--platform linux/amd64`. Thêm vào docker build hoặc set env:

```powershell
$env:DOCKER_DEFAULT_PLATFORM = "linux/amd64"
```

**Container exits with SIGKILL after few seconds**

OOM. Tăng Docker Desktop memory: Settings → Resources → Memory ≥ 6 GB. Restart Docker Desktop.

**`/sections` returns 500 with "Extraction failed"**

Check `docker logs chord-test`. Common:
- `subprocess.CalledProcessError ... <Signals.SIGKILL: 9>`: Demucs subprocess OOM
- `ImportError ... natten.functional`: NATTEN wheel không match torch version (rebuild)
- `ImportError ... MutableSequence from collections`: madmom compat patch không apply (kiểm tra extractor.py có gọi `_patch_madmom_compat()` và `_patch_natten_cpu()` trước `import allin1`)

**yt-dlp download fails 403**

YouTube đôi khi block server IPs. Update yt-dlp trong image bằng cách rebuild với:

```dockerfile
&& /app/.pixi/envs/default/bin/pip install --no-cache-dir -U yt-dlp
```

Thêm vào pip install layer.

### GPU phase

**`docker run --gpus all` fails với "could not select device driver"**

NVIDIA Container Toolkit chưa setup trên Docker Desktop WSL2. Fix:

```bash
# Trong WSL2 distro
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Hoặc enable trong Docker Desktop: Settings → Resources → WSL Integration → enable distro that has nvidia-container-toolkit.

**`torch.cuda.is_available()` returns False inside container**

- Check `nvidia-smi` chạy được từ container chưa: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
- Verify base image trong Dockerfile.gpu là `nvidia/cuda:12.1.0-runtime-ubuntu22.04` (không phải debian)
- Verify torch installed có CUDA: `docker exec chord-test pip show torch | findstr Version` should show `2.5.0+cu121`

**NATTEN CUDA kernels missing — fallback to Triton**

Có thể thấy log: `_IS_TRITON_SUPPORTED = ... < 70` cho 3050 Ti (compute capability 8.6). Triton support cần GPU CC ≥ 7.0; 3050 Ti = 8.6 nên OK. Nếu NATTEN vẫn fall back, kiểm tra triton package có installed: `docker exec chord-test pip show triton`. Nếu thiếu: thêm `triton` vào pip install layer.

**Out of GPU memory (3050 Ti có 4 GB VRAM)**

`torch.OutOfMemoryError: CUDA out of memory`. hdemucs_mmi + single-fold allin1 nên fit trong 4 GB, nhưng nếu fail:
- Đảm bảo không có process khác chiếm VRAM trên Windows (close games, browser hardware accel có thể chiếm 1-2 GB)
- Thử `--device cpu` cho 1 stage (vd: model run CPU, demucs run CUDA): sửa code passes device khác nhau

---

## Reference — code change summary của repo này

3 thay đổi chính so với upstream:

1. **`app/extractor.py`** — thêm `extract_sections()`:
   - Lazy import allin1
   - Apply 2 compat patches trước import: `_patch_madmom_compat()` (Python 3.11 / numpy compat) và `_patch_natten_cpu()` (stub `torch.cuda.get_device_capability` cho CPU torch — không cần trên GPU host)
   - Monkey-patch `allin1.demix.demix` và `allin1.analyze.demix` qua `sys.modules` (vì `allin1/__init__.py` shadows `allin1.analyze` thành function nên `import allin1.analyze` không cho ta module reference)
   - Default model `harmonix-fold0` (single-fold, 8× nhanh hơn ensemble); override qua env `ALLIN1_MODEL=harmonix-all` để dùng ensemble
   - Demucs forced `hdemucs_mmi` (không config được, nếu cần htdemucs phải edit code)
   - Pass tempdir cho `demix_dir`/`spec_dir` để cô lập concurrent requests
   - `multiprocess=False` (tránh fork overhead trong uvicorn threadpool)
   - Drop nhãn `start`/`end` ra khỏi response (silence markers, không có ý nghĩa cấu trúc)

2. **`app/schemas.py`** — `SectionsResponse` shape:
   ```python
   class SectionsResponse(BaseModel):
       duration: float
       bpm: int
       beats: list[float]
       downbeats: list[float]
       segments: list[Section]
   ```

3. **`app/main.py`** — route `POST /sections` (same pattern as existing `/extract`, `/bpm`, `/meter`).

4. **`Dockerfile`** — multi-stage:
   - Build stage: pixi env + pip layer (torch CPU 2.5.0, NATTEN 0.17.4 CPU wheel, allin1, diffq)
   - Prefetch model weights: harmonix-all (8 folds → ~80 MB) + hdemucs_mmi (~80 MB) + htdemucs (~250 MB fallback)
   - Runtime: copy env + cache, env vars `HF_HOME=/opt/cache/huggingface`, `TORCH_HOME=/opt/cache/torch`

Image hiện tại 3.2 GB (CPU). GPU variant dự kiến 6-8 GB.
