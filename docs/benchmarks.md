# Benchmarks

Measured with `scripts/gpu_check.py`, settings `beam_size=5`,
`word_timestamps=True`, `vad_filter=True`. The RTF column is audio seconds per
wall-clock second, so higher is faster. VRAM is the extra memory in use while
the model is loaded, measured with `nvidia-smi`.

## 2026-09-26: RTX 5070 (12 GB, Blackwell sm_120), driver 610.74

Stack: faster-whisper 1.2.1, CTranslate2 4.8.2, cuBLAS 12.9 and cuDNN 9.26
(from pip wheels, see `whisper_subs/gpu.py`), Python 3.11.9.

Clip: `samples/tts_en_check.wav`, 99.3 s of **English** Windows TTS. No
Japanese voice is installed, so this measures speed and whether the GPU works
at all, not Japanese accuracy. The encoder cost doesn't depend on language.
Decoding cost does depend somewhat on tokens per second, so re-measure on real
Japanese samples.

| Model | Compute | Load s | Run s | RTF | VRAM MiB |
|---|---|---:|---:|---:|---:|
| small | float16 | 0.8 | 3.5 | 28.7× | ~800 |
| medium | float16 | 1.2 | 6.1 | 16.3× | ~2,000 |
| large-v3-turbo | float16 | 1.2 | 2.1 | **46.5×** | ~2,100 |
| large-v3-turbo | int8_float16 | 1.8 | 2.5 | 40.3× | ~1,100 |
| large-v3 | float16 | 2.6 | 8.0 | 12.5× | ~4,100 |
| large-v3 | int8_float16 | 3.8 | 10.2 | 9.7× | ~2,100 |

The first run of the day was slower (`small` at 10.3×) because of CUDA
initialisation and cuDNN autotuning, so warm runs are the ones reported.

### What this means

- **The GPU works.** Blackwell runs fine on CTranslate2 4.8.2 once cuBLAS and
  cuDNN are available. There's no need for a CPU fallback on this machine.
- **Speed isn't the constraint.** Even `large-v3` does a 1-hour archive in
  about 5 minutes, and turbo in about 80 s. Pick the model on Japanese
  accuracy, which Milestone 1 measures on real samples.
- **int8 halves the VRAM and costs 15 to 20% speed.** It's only worth using
  when sharing the GPU.
- **Keep one model on the GPU at a time.** With `qwen3.5:9b` loaded in Ollama
  (5.5 GB at its default 4k context), `large-v3` fp16 still fit and ran at
  11.3×, with about 11.2 GB in use out of 12.2 GB. The translation pipeline
  runs Ollama at `num_ctx=16384`, which needs more KV cache than that, so
  running both at once risks Ollama spilling layers to the CPU. Unload Whisper
  before pass 1, as the plan says (Milestone 2.3).

### Gotchas hit during setup

1. `RuntimeError: Library cublas64_12.dll is not found`. CTranslate2's Windows
   wheel doesn't bundle cuBLAS or cuDNN. The fix is
   `pip install -e ".[cuda]"`, and `enable_cuda_dlls()` adds the wheels' `bin`
   directories to the DLL search path.
2. The `large-v3` download hung at 0 bytes through Hugging Face's Xet
   transfer. Setting `HF_HUB_DISABLE_XET=1` fixed it: 2.9 GB in about 6 minutes.
3. Hugging Face symlink warnings on Windows are harmless (the cache just uses
   more disk). Silence them with `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.

## To do

- [ ] Re-run on real Japanese samples (Milestone 0.4). Record the RTF and
      compare accuracy of `large-v3` and `large-v3-turbo`, because turbo is
      reported to lose more on Japanese than on English.
