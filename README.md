<div align="center">

```
  ___  __  __ _____ ____  
 / _ \|  \/  |_   _|  _ \ 
| | | | |\/| | | | | |_) |
| |_| | |  | | | | |  __/ 
 \__\_\_|  |_| |_| |_|    
  L L A M A . C P P
```

**Infrastructure patches for Qwen3.5-27B MTP speculative decoding.**

*End-to-end port of the MTP head in llama.cpp.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

## 📑 Table of Contents
- [🎯 Overview](#-overview)
- [📦 The Patches](#-the-patches)
- [📖 The Journey](#-the-journey)
- [📊 Performance Numbers](#-performance-numbers)
- [🚀 Quick Start](#-quick-start)
- [🔗 Related Repositories](#-related-repositories)
- [📄 License](#-license)

---

## 🎯 Overview

This repository carries the **infrastructure patches** as a clean ordered series to support Multi-Token Prediction (MTP) for Qwen3.5-27B in [llama.cpp](https://github.com/ggerganov/llama.cpp). It acts as the substrate that the optimization variants and research repositories build upon.

---

## 📦 The Patches

| # | Patch | What it does |
|---|---|---|
| 01 | qwen3next MTP graph | Wires `LLM_GRAPH_TYPE_MTP` for the qwen3next architecture |
| 02 | qwen35 MTP graph | Mirrors the qwen3next path for the dense Qwen3.5 family |
| 03 | qwen35 end-to-end load+execute | Converter + loader + tensor classification fixes |
| 04 | mask tensor naming diag | Names `kq_mask` tensors so the ggml scheduler bug surfaces in stack traces |
| 05 | chain `prev_hidden` | Threads the hidden state from each MTP step to the next |
| 06 | private `sched_mtp` | Isolates the MTP graph compute in its own scheduler |
| 07 | host-side rollback v1 | Snapshot + restore for the recurrent half on rejection |
| 08 | AR re-decode + `MTP_FORCE_AR` | Plain-decode-equivalent path for diagnostic baselines |
| 09 | in-graph AR loop | Replaces chunking kernel with a sequential AR loop |
| 10 | batched rollback re-decode | Single T=N `llama_decode` instead of N sequential T=1 calls |
| 11 | **rollback bookkeeping fix** | The one-line cache-bookkeeping fix for correct output |

---

## 📖 The Journey

Qwen3.5-27B is a hybrid architecture: 48 DeltaNet layers interleaved with 16 full-attention layers, and one MTP head as layer 64. 

Getting this working required fixing a multitude of issues from silent tensor stripping to a missing recurrent memory module snapshot/restore primitive. Patch 11 provides the final unblock—a single `id_last = corr` → `id_last = argmax(tail_logits)` correction after a batched rollback re-decode.

---

## 📊 Performance Numbers

On Qwen3.5-27B Q4_K_M, M4 Max (post-fix):

| Path | tok/s | vs plain | Output |
|---|---|---|---|
| Plain decode (`llama-bench tg32`) | **17.90** | 1.00× | ✓ correct |
| K=1 MTP spec (this branch) | **7.64** | 0.43× | ✓ correct |

> [!NOTE]
> Single-MTP-head spec path is currently slower than plain decode on this hybrid model. Optimization variants in [qwen-mtp-optimizations](https://github.com/quivent/qwen-mtp-optimizations) act as the levers to speed this up.

---

## 🚀 Quick Start

### Applying the patches

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
# These patches apply against the upstream commit recorded in patches/00-base.txt
git am path/to/qwen-mtp-llamacpp/patches/*.patch
cmake -B build && cmake --build build -j 12 --target llama-mtp-speculative
```

### Reproducing the benchmark

```bash
MODEL=path/to/qwen3.5-27b-q4km.gguf

# Plain decode (ground truth)
./build/bin/llama-bench -m $MODEL -p 0 -n 32 -ngl 99

# K=1 MTP spec (this branch)
./build/bin/llama-mtp-speculative -m $MODEL \
    -p "Explain photosynthesis in one paragraph." \
    -n 64 -ngl 99 -c 2048
```

---

## 🔗 Related Repositories

- **[qwen-mtp-tensors](https://github.com/quivent/qwen-mtp-tensors)**
- **[qwen-mtp-optimizations](https://github.com/quivent/qwen-mtp-optimizations)**
- **[qwen-mtp-research](https://github.com/quivent/qwen-mtp-research)**

---

## 📄 License

Patches are MIT-licensed (matching upstream llama.cpp).
