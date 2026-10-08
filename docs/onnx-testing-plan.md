# ONNX Test Coverage & Cross-Backend Parity Plan

Status: T1–T5 implemented (2026-10-08). Coverage tests: 7/7 green ort-only.
Parity tests: 5 active and green; 8 `#[ignore]`d pending the beam-search scoring
investigation below.

## 1. Goals

1. **Coverage parity**: every pipeline *behavior* covered by a libtorch test has an ONNX counterpart, wherever the same (or equivalent) checkpoint exists as an ONNX export on the hub.
2. **Numerical parity**: for the same checkpoint and input, the `tch` and `ort` backends produce the same outputs within a defined tolerance — verified in a single test process (dual `libtorch + onnx` build), not via two drifting sets of golden values.
3. **Architecture validation (stretch)**: where an official ONNX export exists, use it as ground truth to validate the hand-written Rust model ports.
4. All ONNX tests run in the **ort-only CI job** (no libtorch); parity tests run in a separate **dual-backend job**.

## 2. Current state

- 123 test functions total: **117 libtorch** (28 files) + **6 ONNX** (`tests/onnx.rs`).
- ONNX coverage today: masked LM (BERT), QA (RoBERTa), sentiment (DistilBert), NER full entities (BERT), greedy generation (GPT2, with past), translation (M2M100, with past).
- **ONNX code paths with zero test coverage**: conversation pipeline, sentence embeddings (ONNX encoder + ndarray pooling), keywords extraction, zero-shot classification, summarization, all generation variants except greedy, and the decoder-without-past fallback.

## 3. Test-category policy

Each libtorch test maps to exactly one category:

| Category | Examples | ONNX counterpart | Parity assertion |
|---|---|---|---|
| **A. Pipeline behavior** | sentiment, NER, QA, masked LM, summarization output, translation output | Port the test with `ModelResource::ONNX` | Final outputs exact (labels/strings); scores `atol 1e-4` |
| **B. Generation variants** | beam search, diverse beam, bad tokens, prefix-allowed, token scores, padding | Port; greedy + beam are deterministic | Exact generated text for greedy/beam; token scores `atol 1e-3` |
| **C. Sampling** | top-k/top-p sampling tests | Port with fixed seed | Compare last-step logits, **not** sampled tokens (ulp-level logit differences flip RNG draws) |
| **D. Architecture numerics** | `gpt2_lm_model`, `bert_masked_lm` logit assertions | Stretch (Phase T6): run raw ONNX session vs Rust model on same input | Logits `atol 1e-4, rtol 1e-3` |
| **E. Backend internals** | `.ot` loading, `kind`/half casting, device placement, hidden-states/attentions outputs | **Out of scope** (no ONNX analog; standard exports don't expose intermediates) | — |

Tolerance policy (f32, CPU): logits/scores/embeddings `atol 1e-4`, `rtol 1e-3`; embeddings compared via cosine similarity > 0.999; final tokens/text exact for deterministic decoding.

## 4. Model availability matrix

Checkpoints referenced by existing tests, and the ONNX export situation:

| Checkpoint (libtorch test) | ONNX export | Status |
|---|---|---|
| `bert-base-uncased` (masked LM) | `optimum/bert-base-uncased-for-masked-lm` | ✅ verified, in use |
| `bert-base-NER` (NER) | `optimum/bert-base-NER` | ✅ verified, in use |
| `roberta-base-squad2` (QA) | `optimum/roberta-base-squad2` | ✅ verified, in use |
| `distilbert-base-uncased-finetuned-sst-2-english` | `optimum/distilbert-base-uncased-finetuned-sst-2-english` | ✅ verified, in use |
| `gpt2` (generation) | `optimum/gpt2` (decoder + decoder_with_past) | ✅ verified, in use |
| `m2m100_418M` (translation) | `optimum/m2m100_418M` | ✅ in use (pin commit `e775f50e`) |
| `all-MiniLM-L6-v2` (sentence embeddings) | `optimum/all-MiniLM-L6-v2` | ✅ verified (`onnx/model.onnx`) |
| `t5-small` (translation/summarization) | `optimum/t5-small` | ✅ verified (encoder + decoder + decoder_with_past) |
| `distilbert-base-uncased-mnli` (zero-shot) | `optimum/distilbert-base-uncased-mnli` | ✅ verified |
| DialoGPT (conversation) | community exports exist | ⚠️ to verify (`optimum/dialogpt-medium` or fallback) |
| `distilgpt2` | `optimum/distilgpt2` | ⚠️ to verify |
| `distilbert-base-cased-distilled-squad` (QA) | optimum equivalent | ⚠️ to verify |
| `xlm-roberta` German NER | optimum equivalent | ⚠️ to verify |
| `mobilebert-uncased` / POS model | optimum equivalent | ⚠️ to verify |
| albert, deberta(v2), electra, fnet, longformer, xlnet, reformer, prophetnet, gpt_neo, gpt_j, openai_gpt, longt5, mbart, marian, nllb, pegasus | none official | ❌ Phase T7 (CI self-export) or skip |

All remote ONNX resources must pin a **commit revision** (as the M2M100 test already does), so golden values cannot drift when HF repos are updated.

## 5. Phases

### Phase T1 — Coverage completion (ort-only, ~7 tests, no new infra)

Close the five untested ONNX pipeline paths + the no-past fallback. All gated `#[cfg(feature = "onnx")]`, all run in the existing `test-onnx-only` CI job.

1. `onnx_sentence_embeddings` — `optimum/all-MiniLM-L6-v2`; encode two sentences, assert golden embeddings `atol 1e-4`; also asserts `get_embedding_dim`.
2. `onnx_keywords_extraction` — same checkpoint; cosine-similarity scorer, assert golden keyword set/scores (mirrors `keyword_extraction_cosine_similarity`).
3. `onnx_keywords_extraction_mmr` — maximal-marginal-relevance variant (mirrors `keyword_extraction_maximal_margin_relevance`).
4. `onnx_conversation` — DialoGPT ONNX export; two-turn conversation, assert response strings.
5. `onnx_zero_shot_classification` — `optimum/distilbert-base-uncased-mnli`; mirror the libtorch zero-shot fixture, assert labels + scores.
6. `onnx_summarization` — `optimum/t5-small` (or distilbart if verified); assert summary text.
7. `onnx_text_generation_without_past` — GPT2 decoder-only resource (no `decoder_with_past`), short greedy generation, assert text; proves the fallback cache path.

**Un-gate `tests/sentence_embeddings.rs` from libtorch-only** is explicitly *not* part of this: that file uses `.ot` presets; the ONNX tests live in the onnx test files.

### Phase T2 — Parity harness (infra, no new model tests)

1. New `tests/onnx_parity.rs`, gated `#[cfg(all(feature = "libtorch", feature = "onnx"))]`.
2. Shared helper module providing:
   - `assert_close(array: &ArrayD<f32>, tensor: &Tensor, atol, rtol)` — the tch↔ndarray bridge for comparisons;
   - builders that construct the torch config (`ModelResource::Torch` + preset) and the ONNX config (`ModelResource::ONNX` + pinned export) for the *same checkpoint*, sharing tokenizer resources.
3. New CI job `test-onnx-parity` (libtorch + onnx, `ORT_DYLIB_PATH` set, runs `--test onnx_parity`).

### Phase T3 — Encoder parity (same checkpoint, both backends, ~5 tests)

Each test loads both backends, feeds identical tokenized inputs, and compares:

1. `parity_bert_masked_lm` — logits for masked positions + final predictions.
2. `parity_distilbert_sentiment` — logits + polarity labels.
3. `parity_roberta_qa` — start/end logits + decoded answer.
4. `parity_bert_ner` — token logits + aggregated entities.
5. `parity_sentence_embeddings` — MiniLM embeddings, cosine > 0.999 both directions.

Note: the torch side uses the rust-bert-converted `.ot` weights and the ONNX side the optimum export **of the same upstream checkpoint** — identical weights, so divergences are real findings.

### Phase T4 — Generation parity (~6 tests)

1. `parity_gpt2_greedy` — exact text equality.
2. `parity_gpt2_beam_search` — exact text equality.
3. `parity_gpt2_beam_search_multi_prompt_padding` — exact text, exercises the shared ndarray driver's padding path against the ONNX session cache.
4. `onnx_gpt2_bad_tokens` — port (golden asserts, ort-only).
5. `onnx_gpt2_prefix_allowed` — port (uses the `&[i64]` signature).
6. `onnx_gpt2_token_scores` — port; scores `atol 1e-3`.
7. Sampling (`onnx_gpt2_sampling`) — fixed-seed port asserting last-step logits and *plausibility* of text; **no cross-backend token equality** (category C policy).

### Phase T5 — Encoder-decoder parity (~3 tests)

1. `parity_m2m100_translation` — exact output text for the existing EN→FR/ES/HI fixture.
2. `parity_t5_translation` — `optimum/t5-small`, exact text.
3. `parity_t5_summarization` — exact text (or BLEU-adjacent tolerance if ulp flips surface; decide when observed).

### Phase T6 (stretch) — Architecture validation via ONNX ground truth

For checkpoints with official exports (BERT, DistilBert, RoBERTa, GPT2, T5, M2M100): low-level tests that run the **raw ONNX session** (not the pipeline) and the **Rust model** on identical inputs, comparing logits. This validates the Rust ports against the reference export. Expect real findings; timebox investigation of any mismatch to half a day each.

### Phase T7 (optional) — CI self-export for exotic architectures

Extend the existing Python `convert-model` CI job with `optimum-cli export onnx` for architectures without hub exports (xlnet, reformer, longformer, electra, gpt_neo, …), then port their category-A tests using `LocalResource`. High cost (Python env, export churn, per-arch op support in old onnxruntime versions), low urgency — recommend deferring until T1–T5 are green.

## 6. Full libtorch → ONNX mapping

Legend: **port** = category-A/B port with golden values; **parity** = dual-backend comparison; **skip** = out of scope (category E); **T7** = deferred pending self-export.

| libtorch test file (tests) | Decision |
|---|---|
| `bert.rs` masked_lm, seq_cls, token_cls, QA, NER, NER-full | port + parity (T1/T3) |
| `bert.rs` multiple_choice | skip (no export for MC head) |
| `bert.rs` `*_lm_model` numerics | T6 |
| `distilbert.rs` all 5 | port + parity (sst2 verified; squad to verify) |
| `roberta.rs` masked_lm, seq_cls, token_cls, QA | port + parity (squad2 verified; base exports to verify) |
| `roberta.rs` xlm_roberta_german_ner | port (export to verify) |
| `gpt2.rs` 16 generation tests | port all except `lm_model`; parity for greedy/beam/padding (T4); `lm_model` → T6 |
| `distilgpt2.rs` | port (export to verify), else T7 |
| `openai_gpt.rs`, `gpt_neo.rs`, `gpt_j.rs` | T7 |
| `t5.rs` translation, summarization | port + parity (T5) |
| `longt5.rs`, `pegasus.rs`, `prophetnet.rs` | T7 |
| `bart.rs` summarization greedy/beam | T7 (distilbart export unverified) |
| `marian.rs`, `mbart.rs`, `nllb.rs` | T7 |
| `m2m100.rs` translation | port ✅ + parity (T5); `lm_model` → T6 |
| `sentence_embeddings.rs` sbert ×6 | port MiniLM only (T1); others T7 |
| `sentence_embeddings.rs` keywords ×4 | port 2 representative (T1); remaining 2 follow same code path |
| `conversation` (no dedicated libtorch test file) | new ONNX test (T1) — path shipped in Phase 4 untested |
| `zero_shot` (covered via deberta MNLI test) | new ONNX test with distilbert-mnli (T1) |
| `mobilebert.rs` all 6 | port if export verified, else T7 |
| `albert.rs`, `deberta.rs`, `deberta_v2.rs`, `fnet.rs`, `longformer.rs`, `xlnet.rs`, `electra.rs`, `reformer.rs` | T7 |
| `hf_tokenizers.rs` | orthogonal (tokenizer feature); optional ONNX variants |

## 7. CI wiring

- `test-onnx-only`: extend to run all new `onnx*` test files (T1, T4 ports).
- `test-onnx-parity` (new): default features + `onnx`, `ORT_DYLIB_PATH` set, `--test onnx_parity`.
- Existing libtorch batches unchanged.
- Keep onnxruntime 1.20.1 pinned in CI to match the dev-dep API level (17).

## 8. Risks

- **Finding real divergences** between Rust ports and exports (T3/T6) — the point of the exercise, but turns test-writing into debugging; timebox each.
- **Golden-value drift** — mitigate by pinning commit revisions on every ONNX remote resource.
- **Exact-text parity fragility** — greedy/beam fixtures are strong-confidence; if a boundary flip appears, relax that one case to score tolerance and document it.
- **Hub availability** — ⚠️ rows in §4 must be verified before their phase starts; fallback is skipping or T7.
- **CI runtime growth** — prefer small checkpoints (t5-small, MiniLM, distil*); parity jobs download both formats of each checkpoint (cache mitigates).

## 9. Suggested order & estimates

| Phase | Content | Est. |
|---|---|---|
| T1 | 7 coverage tests | ~1 session |
| T2 | parity harness + CI job | ~½ session |
| T3 | 5 encoder parity tests | ~1 session |
| T4 | 7 generation tests | ~1 session |
| T5 | 3 encoder-decoder parity tests | ~½ session |
| T6 | stretch, per-architecture | open-ended |
| T7 | deferred | — |


## 10. Implementation findings (2026-10-08)

The coverage and parity work surfaced six defects, four of them pre-existing on
`main` and invisible because the torch integration tests were not part of the
phase verification loops:

1. **Inverted attention mask** in `SentenceEmbeddingsPipeline::tokenize_arrays`
   (padded positions marked attended, real tokens masked) — corrupted all
   batched sentence-embedding outputs whenever padding was required. Fixed;
   libtorch sbert goldens pass again and the ONNX MiniLM embeddings now match
   the published reference values.
2. **Row/column swap** in the keywords-extraction MMR / max-sum scorers
   (`column(0)` on a (1, n) matrix) — panicked for n-gram candidates > 1.
3. **T5 configuration unreachable ort-only** — `ConfigOption::T5` and its
   `from_file` mapping were libtorch-gated; extracted a tch-free
   `t5::config` module and un-gated the variant (fixes ONNX T5 summarization).
4. **NLI label order** — the zero-shot ONNX path assumed contradiction=0 /
   entailment=2; now read from the model config `id2label`.
5. **Position-id off-by-one** in decoder-only `prepare_inputs`
   (`max(cumulative - 1, 1)` instead of `..., 0)`) — shifted all positions by
   one, breaking torch-side greedy generation (fixed in GPT2/GPT-J/GPT-Neo/
   Reformer and the ONNX generator; ONNX goldens were self-consistent with the
   bug and must be re-captured when the beam work lands).
6. **Rotated tensor conversions** in the `forward_t` of the T5, BART, MBart,
   M2M100, LongT5 and ProphetNet generators (encoder_outputs bound to
   input_embeds, decoder_input_ids to encoder_outputs, ...) — made torch-side
   encoder-decoder generation panic or produce garbage.

**Open follow-up (top priority):** beam-search scoring in the shared ndarray
driver diverges from the pre-refactor goldens (torch-side; ONNX beam search
matches its own goldens). Affected: `tests/gpt2.rs` beam tests, torch T5/BART/
Marian/MBart/M2M100 summarization & translation goldens, and 8 `#[ignore]`d
parity tests in `tests/onnx_parity.rs`. Also: NaN per-token scores when a
`prefix_allowed_tokens_fn` is used (both backends).
