# Agent latency optimization design

This document summarizes measurement-driven latency work for the Mercari shopper agent. It records what we changed, what we deliberately did **not** change (rate limits, Mercari 429 behavior, quality risk), and how to verify results.

## Problem

End-to-end agent runs were ~30–37s for a typical three-step flow (search → details → submit). The original hypothesis was browser/Playwright fallback cost; **Phase A measurement disproved that** for the DPoP API path: HTTP 200 throughout, no `source_fallback`, and Playwright is often not installed.

Measured cost split (verbose `StepTrace` + structured logs):

| Area | Share of wall time (baseline) | Notes |
|------|-------------------------------|--------|
| LLM round-trips (3 turns) | ~56% | Serial; large system + tools prefix every turn |
| `get_item_details` (serial, up to 10 ids) | ~38% | One HTTP call per id through a fully serial rate limiter |
| Enrich / other | Small | Seller rating fetch when list row lacks rating |

## Constraints (why not “optimize everything”)

1. **Mercari rate limits** — 401/403/429 on the API path triggers `AutoSource` sticky fallback to `BrowserSource` (slower; may be unavailable). Aggressive parallelism or zero gap between requests increases 429 risk.
2. **Average request spacing** — We keep a configurable minimum interval between request *starts* (`MERCARI_MIN_INTERVAL`, jitter) while allowing a **bounded burst** via concurrency.
3. **Quality / grounding** — Fewer detail ids and shorter `reason` text can affect recommendations; full golden-query eval is the gate before riskier prompt changes.
4. **Provider caching** — Anthropic prompt caching requires a minimum cacheable prefix (512 tokens for Claude Haiku 5.5). Our system + tools block is ~2.6k tokens; below the minimum, caching is silently skipped.

## Phase A — Measurement only (no behavior change)

**Goal:** Confirm where time goes; do not change agent logic.

| Change | Purpose |
|--------|---------|
| `ToolExecution.elapsed_s` + `tool_call` log field | Per-tool wall time |
| `StepTrace`: `llm_s`, `tools_s`, `tool_elapsed_s` | Per-step LLM vs tools split in `--verbose` |
| `llm_complete.elapsed_s` | Per LLM call duration |
| Enricher `enrich` / `enrich_seller` timing | Optional seller-rating cost |
| Suppress noisy SDK loggers at WARNING even when `--verbose` | Readable JSON trace |

**Finding:** Dominant costs are serial detail fetches and three LLM turns—not browser launch.

## Phase B — Source layer (rate limit + concurrency)

**Goal:** Overlap detail fetches without removing average spacing.

### Rate limiter redesign (`mercari_shopper/sources/retry.py`)

- **Before:** `asyncio.Lock` held through `sleep` → concurrent waiters were fully serialized.
- **After:** Reserve a start slot under the lock, release lock, then sleep. A deque of the last `max_concurrent` start timestamps enforces minimum gap per “lane.”
- **`max_concurrent=1`:** Behavior matches the old serial limiter (tests preserved).

### Bounded concurrency

- `Settings.max_concurrent` / env `MERCARI_MAX_CONCURRENT` (default **2**).
- `DpopApiSource.get_items` and `BrowserSource.get_items`: `Semaphore(max_concurrent)` + `asyncio.gather`, order preserved in results.

**Risk posture:** Two in flight with interval kept—aligned with plan decision “allow 2 in flight, with the gap kept,” not unbounded parallel scraping.

## Phase C — Agent / schema trimming (low risk)

| Change | Default / value | Rationale |
|--------|-----------------|-----------|
| `MAX_DETAIL_IDS` | **7** (was 10; briefly 5 in development) | Cuts detail HTTP count; prompt text updated to match |
| `RecommendationArgs.reason` max length | **300** (was 800) | Smaller submit payload; fewer validation retries on oversized reasons |

**Not done in this PR (higher risk):**

- **C5 — Collapse 3 LLM turns → 2** (e.g. batch market price in turn 1, skip details when search rows are sufficient). Highest latency upside; requires eval harness before merge.
- **Seller rating enrich** — Kept; only when list row did not already include rating (plan: keep, avoid duplicate fetches).

## Phase C8 — Prompt caching (Anthropic / Vertex)

**Goal:** Reduce input processing on turns 2+ for the static system + tools prefix.

- `AnthropicAdapter.build_payload`: system as a single text block with `cache_control: { type: ephemeral }`; same on the **last** tool in `tools` (caches everything up through that block).
- `AnthropicVertexAdapter` inherits `build_payload` — Vertex Haiku 5.5 benefits without a separate implementation.
- `LLMResponse` + `llm_complete` log: `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens` for verification.

**Live check (Vertex, 3-step run):** Turn 1–2 may `cache_write` ~2647 tokens; turn 3 showed `cache_read_tokens: 2647`. Rapid back-to-back turns can miss read on turn 2 (provider propagation); hits improve within the 5-minute TTL.

OpenAI path unchanged (server-side caching where supported; token fields stay 0).

## Results (indicative live runs, same query family)

| Configuration | Approx. `agent_complete.elapsed_s` | Notes |
|---------------|-------------------------------------|--------|
| Baseline (10 ids, serial limiter) | ~30.2 | Phase A baseline |
| Cap 5 + concurrency 2 | ~24.6 (extra submit validation turn ~21s “clean”) | Incidental schema retry, not grounding |
| Cap 7 + concurrency 2 + caching | ~16.4 | 3 steps, 0 repairs |

Detail-step tool time tracked id count and concurrency as expected (e.g. ~11s → ~1.6s for seven ids with two lanes).

## Configuration reference

```env
# Existing spacing between Mercari request starts (with jitter)
MERCARI_MIN_INTERVAL=...

# Max concurrent in-flight Mercari requests (default 2)
MERCARI_MAX_CONCURRENT=2
```

See `.env.example` for comments.

## Verification

```bash
# Unit + integration
python -m pytest tests/unit tests/integration -q
python -m ruff check mercari_shopper tests
python -m mypy mercari_shopper

# Live timing + cache fields (requires credentials)
python -m mercari_shopper --plain --verbose --non-interactive --provider vertex -q "your query"
# Inspect stderr JSON: llm_complete (cache_read_tokens), step trace, agent_complete
```

**Phase D (recommended next):** Run `python -m mercari_shopper.eval` on golden queries to confirm no quality regression from cap 7 and reason length 300.

## Out of scope / follow-ups

- Fair-value scoring anomalies on some listings (quality, not latency).
- Distinct verbose logging for submit **validation** errors vs **grounding** failures (operability).
- Phase C5 turn collapse after eval sign-off.
- Tuning `MERCARI_MAX_CONCURRENT` above 2 without monitoring 429 and `source_fallback` rates.
