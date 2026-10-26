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

## Indexer CP with multi-CTA top-k

```python
built.forward(plan, input_ids, positions, forward_batch, indexer_cp=True)
```

Every TP rank must pass the same Python bool. The default is `False`. Both
indexer variants are built before forward; CP does not lazily compile a
variant. A captured graph retains its CP choice. Callers supporting both modes
must retain separate graphs and include CP in their graph-cache key. Ordinary
SGLang server calls omit this keyword and retain the default.

For direct layer calls, prepare matching base and CP runtimes collectively:

```python
op = Glm53LayerOp(base_runtime, weights, layer, cp_runtime=cp_runtime)
out = op.forward(h, caches, cos, sin, step, indexer_cp=True)
```

Advance only the selected runtime after all of its layer calls in a step.
The plugin forward handles this itself. Reuse-only layers use their original
runtime in both modes. Each runtime has its own workspace, peer buffers and
epoch; weights and caches remain shared.

Both modes use upstream [#42](https://github.com/sgl-project/sgl-amd-mono/pull/42)'s
wave scoring. CP partitions each row's live 64-key tiles across ranks. Empty
shards require no score work. Compiled contexts up to 16K also shard index-Q.
TP8 width two uses two CP4 replicas at compiled capacities in (16K, 128K].

| Live keys per row | CP selection |
| --- | --- |
| Up to 32K | Exchange scores within the CP group; use the single-CTA selector |
| Above 32K through 128K | Exchange scores; use upstream [#24](https://github.com/sgl-project/sgl-amd-mono/pull/24)'s multi-CTA histogram/gather selector |
| Above 128K | Keep scores in rank-local shards; use multi-CTA distributed radix selection and exchange summaries and final indices |

Local selection uses up to 16 CTA slices per row. Wide/cohort grids cap the
slice count to fit their resident upper half. Long CP selection retains four
256-bin radix rounds and the local selector's 12/24-bit candidate ordering,
including threshold ties. The result is exact top-2048 in compact physical-slot
form; its output order is not a score sort. Live lengths can cross these
thresholds during graph replay. CP does not distribute cache storage, and its
latency benefit depends on the workload.

Peer buffers must be closed collectively before destroying the process group.
All ranks finish unmapping before an exporter frees its storage. A failed close
retains ownership for collective retry; repeated successful closes are safe.

## Correctness checks

Run on four available gfx950 GPUs in the project's FlyDSL/AITER environment:

```bash
python -m pytest -q test/unit/test_glm53_spec.py test/unit/test_glm53_select.py \
  test/unit/test_glm53_forward.py test/unit/test_glm53_forward_spec.py \
  test/unit/test_glm53_index_plan.py test/unit/test_glm53_mxfp4.py \
  test/unit/test_glm53_score_check.py test/unit/test_peer.py

python -m pytest -q test/kernels/test_glm53_kernels.py -k 'not tp8'
```

The GPU checks cover CP equivalence, independent score arithmetic, independent
A4W4 MoE arithmetic and quantization codes/scales, serial/batched state,
causality, cache preservation, padding, replay and HIP peer lifetime. They use
synthetic fixtures. Full A4W4 checkpoint quality, multilayer quality, MTP
acceptance and serving mono-kernel on/off results require serving validation.
