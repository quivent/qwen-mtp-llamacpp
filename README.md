# qwen-mtp-llamacpp

> Multi-Token Prediction (MTP) speculative decoding for **Qwen3.5-27B** in [llama.cpp](https://github.com/ggerganov/llama.cpp). End-to-end port of the MTP head from HuggingFace → GGUF → loader → graph builder → speculative decoding loop, on a hybrid attention + DeltaNet architecture.

This repo carries the **infrastructure patches** as a clean ordered series. It is the substrate that the [qwen-mtp-optimizations](https://github.com/quivent/qwen-mtp-optimizations) and [qwen-mtp-research](https://github.com/quivent/qwen-mtp-research) repos build on top of.

## What's in here

| # | Patch | What it does |
|---|---|---|
| 01 | qwen3next MTP graph | Wires `LLM_GRAPH_TYPE_MTP` for the qwen3next architecture |
| 02 | qwen35 MTP graph | Mirrors the qwen3next path for the dense Qwen3.5 family |
| 03 | qwen35 end-to-end load+execute | Converter + loader + tensor classification fixes (5 separate fixes in one commit) |
| 04 | mask tensor naming diag | Names `kq_mask` tensors so the ggml scheduler bug surfaces in stack traces |
| 05 | chain `prev_hidden` across K draft steps | Threads the hidden state from each MTP step to the next |
| 06 | private `sched_mtp` | Isolates the MTP graph compute in its own scheduler — fixes a cross-decode state leak |
| 07 | host-side rollback v1 | Snapshot + restore for the recurrent half on rejection; `seq_rm_attn_only` for the attn half |
| 08 | AR re-decode + `MTP_FORCE_AR` | Plain-decode-equivalent path for diagnostic baselines |
| 09 | in-graph AR loop for T≤16 verify | Replaces the chunking DeltaNet kernel with a sequential AR loop inside one graph dispatch |
| 10 | batched rollback re-decode | Single T=N `llama_decode` instead of N sequential T=1 calls |
| 11 | **rollback bookkeeping fix** | The one-line cache-bookkeeping fix that makes the entire spec path produce correct output |

## The journey, in one paragraph

Qwen3.5-27B is a hybrid architecture: 48 DeltaNet (linear-attention recurrent) layers interleaved with 16 full-attention layers, and one MTP head as layer 64. llama.cpp's converter silently strips MTP tensors, the loader doesn't know about `nextn_predict_layers`, the tensor classifier puts MTP tensors in the wrong layer category, the graph builder for QWEN35 doesn't have a draft path, and the recurrent memory module's snapshot/restore primitive doesn't exist. Each of those is a fix on the way from "the model loads" to "the spec path produces text identical to plain decode." Patch 11 is the final unblock: a single `id_last = corr` → `id_last = argmax(tail_logits)` after a batched rollback re-decode, which had been silently double-writing the correction token into the cache and corrupting every position downstream by one slot.

## Honest performance numbers (post-fix)

On Qwen3.5-27B Q4_K_M, M4 Max:

| Path | tok/s | vs plain | Output |
|---|---|---|---|
| Plain decode (`llama-bench tg32`) | **17.90** | 1.00× | ✓ correct |
| K=1 MTP spec (this branch) | **7.64** | 0.43× | ✓ correct |

The infra is correct. The single-MTP-head spec path is currently slower than plain decode on this hybrid model because the MTP draft pass costs about as much as a main forward pass — single-head spec doesn't yet beat plain decode here. The optimization variants in [qwen-mtp-optimizations](https://github.com/quivent/qwen-mtp-optimizations) and the per-position-heads design in [qwen-mtp-research](https://github.com/quivent/qwen-mtp-research) are the next levers.

## Applying the patches

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
# These patches apply against the upstream commit recorded in patches/00-base.txt
git am path/to/qwen-mtp-llamacpp/patches/*.patch
cmake -B build && cmake --build build -j 12 --target llama-mtp-speculative
```

## Reproducing the benchmark

```bash
MODEL=path/to/qwen3.5-27b-q4km.gguf

# Plain decode (ground truth)
./build/bin/llama-bench -m $MODEL -p 0 -n 32 -ngl 99

# K=1 MTP spec (this branch)
./build/bin/llama-mtp-speculative -m $MODEL \
    -p "Explain photosynthesis in one paragraph." \
    -n 64 -ngl 99 -c 2048
```

## Related repos

- **[qwen-mtp-tensors](https://github.com/quivent/qwen-mtp-tensors)** — the converter + tensor-naming work in isolation, with a deep dive on how Qwen3.5's MTP layer is laid out in HuggingFace weights vs GGUF
- **[qwen-mtp-optimizations](https://github.com/quivent/qwen-mtp-optimizations)** — six speculative-decoding variants explored on top of this infra (adaptive chain, predictive hidden, perturbed ensemble, branching tree, drift refresh, ensemble fast-path)
- **[qwen-mtp-research](https://github.com/quivent/qwen-mtp-research)** — research notes, methodology, the hybrid-attention/DeltaNet/MTP interaction map, and the per-position MTP heads design (DeepSeek V3 style)

## License

Patches are MIT-licensed (matching upstream llama.cpp). The patches contain modifications to llama.cpp source files; the original copyright notices in those files apply. See [LICENSE](LICENSE).
