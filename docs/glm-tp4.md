# GLM-5.3 at TP4

The GLM plugin runs the 75 MoE layers in one launch per layer, with up to 16
query tokens per step at TP4. Dense layers, the final norm and logits use
SGLang's existing forward. Larger steps use the model fallback. TP8 retains
its existing eight-row range; the additional A4W4 path requires TP4.

The grouped/cohort schedules are adapted from [ROCm/FlyDSL#1256](https://github.com/ROCm/FlyDSL/pull/1256)
at `032072c9406475cde47cf2017506f7c2604d5cce`, using SGLang's request, weight and
paged-cache layouts.

## Weights and caches

| Loaded experts | MoE activations | Requirements |
| --- | --- | --- |
| Block-FP8 | FP8 (A8W8) | AITER shuffled weights, FP32 block-128 scales |
| Quark/AITER MXFP4 | FP4 (A4W4) | TP4, E2M1 weights and E8M0 scales, separated gate/up rows |

A4W4 reads `QuarkW4A4MXFp4MoE` storage in place: 256 routed experts and one
fused shared expert, local intermediate width 512. Weight shapes are
`[257, 1024, 3072]` and `[257, 6144, 256]`; shuffled scale shapes are
`[257, 1024, 192]` and `[257, 6144, 16]`. Attention and indexer projections
still require the existing shuffled block-FP8 format.

The loader rejects ambiguous packed bytes, W4A8, GU-interleaved/SiTU storage,
missing scales and unshuffled weights. Loaded tensors determine the activation
format. An explicit `moe_activation_dtype="fp8"` or `"fp4"` forward argument must
match those tensors; forward does not convert weights.

MLA cache supports contiguous BF16 or SGLang's raw E4M3 HIP layout: 576 values
per physical slot. Scaled E4M3 rows from other DSA backends are incompatible.
The index cache remains shuffled paged FP8 with 64-token pages and one FP32
power-of-two scale per token. Cache dtype is chosen at build time from SGLang.
The wide schedule requires a full 256-CU MI355X.

## Widths and multi-token steps

A8W8 uses buckets 2, 4, 6, 8, 12 and 16. A4W4 uses exact widths 1–12 and
pads 13–16 logical rows to the 16-row cohort bucket. The direct native builder
also accepts every physical width 1–16.

Verify steps must contain one linear draft chain per request: EAGLE top-k 1
and draft tokens equal to draft steps plus one. Each query sees keys only
through its own causal position, retains its request's page table, and writes
its assigned physical slot. Padding writes no cache and attends no keys.
Auxiliary hidden-state capture uses the unpadded input before launch outputs
reuse the buffer.

## Multi-CTA top-k

Indexer scoring uses upstream [#42](https://github.com/sgl-project/sgl-amd-mono/pull/42)'s
wave scoring on every TP rank. Rows above 32K live keys use upstream
[#24](https://github.com/sgl-project/sgl-amd-mono/pull/24)'s multi-CTA
histogram/gather selector; shorter rows use a single-CTA selector.

The selector uses up to 16 CTA slices per row. Wide/cohort grids cap the slice
count to fit their resident upper half. The result is exact top-2048 in compact
physical-slot form, with the upstream threshold-tie and candidate ordering;
its output order is not a score sort. New keys are read from the complete
launch's mailboxes across row cohorts. Live lengths may cross the selector
threshold during graph replay.

Peer buffers must be closed collectively before destroying the process group.
All ranks finish unmapping before an exporter frees its storage. A failed close
retains ownership for collective retry; repeated successful closes are safe.

## Correctness checks

Run on four available gfx950 GPUs in the project's FlyDSL/AITER environment:

```bash
python -m pytest -q test/unit/test_glm53_spec.py test/unit/test_glm53_select.py \
  test/unit/test_peer.py

python -m pytest -q test/kernels/test_glm53_kernels.py::test_mxfp4_tp4
```

The single parameterized GPU test covers 1×16, 2×8 and 4×4 steps: independent
A4W4 MoE arithmetic and quantization codes/scales, serial/batched state,
expanded page tables, causality, cache preservation and replay. Its fixture,
independent arithmetic oracle and serial control are shared helpers. It uses
128K live contexts in the 4×4 arithmetic case and 64-slot caches in the
expanded-page-table regression. It uses synthetic weights without a server
profile or checkpoint conversion. Full A4W4 checkpoint quality, multilayer
quality, MTP acceptance and serving
mono-kernel on/off results require serving validation.
