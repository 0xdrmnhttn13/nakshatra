# nakshatra

A minimal from-scratch Gemma 3 1B inference engine for Apple Silicon (C++20 CPU reference first, Metal after), with a small compiler track: typed graph IR, verifier, pass pipeline, and lowering.

See `docs/` (separate private repo) for the full codebook.

## TODO / roadmap

### Ordered path to first token, then Metal
- [x] 05 — safetensors loader (DONE: header + impl + FetchContent nlohmann/json + roundtrip exercise)
- [x] 06 — one Gemma decoder block (DONE: random-weight run, 1152->1152, finite)
- [x] 09 — integration skeleton (DONE: decode_one + generation loop runs end-to-end on toy config, random weights)
- [ ] weight glue: map HF tensor names (model.layers.N...) -> ModelWeights structs, load real Gemma weights
- [ ] download google/gemma-3-1b-pt into models/ (gated: accept license, HF token) + token IDs via tools/dump_tokens.py
- [ ] first real token end-to-end (parity target vs reference)
- [ ] 08 — benchmark baseline on CPU (p50/p95/p99 us/token, phase-separated)
- [ ] 06A CPU half — lower Graph → ExecutionPlan, executor dispatching to ch02/03 kernels; equivalence test: handwritten decoder vs graph-lowered, numbers must match
- [ ] 07 — Metal kernels (scale, matvec_naive, rmsnorm_serial)
- [ ] 06A Metal half — lower(Target::Metal), explicit legality, no silent CPU fallback
- [ ] 08 — re-benchmark: before vs after Metal, same tests green

### KV cache upgrades (beyond the naive full cache)
- [ ] hybrid cache: sliding-window ring buffer for local layers (512), full cache for global layers — chapter 04 "month-2 upgrade"
- [ ] KV cache quantization: Q8 / Q4 (llama.cpp `--cache-type-k/v`, ds4-style asymmetric recipe)
- [ ] block/paged cache: fixed-size blocks + position→block table (vLLM PagedAttention style), kill max-capacity reservation
- [ ] prefix caching: reuse KV of shared prefixes across requests (vLLM / SGLang radix tree)
- [ ] study-only: MLA (DeepSeek) — model-side KV compression; StreamingLLM attention sinks

### Compiler track (after 06A works end to end)
- [ ] tiny MLIR dialect: 2-3 Nakshatra ops, one ported lowering pattern (compare architectures, do not restart project in MLIR)
- [ ] buffer liveness analysis for ExecutionPlan reuse (def index / last-use / bytes)
