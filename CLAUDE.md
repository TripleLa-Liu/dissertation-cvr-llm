# CLAUDE.md

## Running Python scripts

Always use the project's `uv` environment, never the system Python or `pip`:

```
uv run --project /Users/triplela0213/Developer/Projects/dissertation-cvr-llm python <script>.py
```

Do not `pip install` into system Python, and do not invoke `python3 <script>.py` directly for anything in this repo (`src/`, `data/preprocessing/`, etc.).

## HuggingFace downloads

Always set `HF_HUB_DISABLE_XET=1` when running anything that downloads a model via `sentence-transformers` / `huggingface_hub` in this environment:

```
HF_HUB_DISABLE_XET=1 uv run --project /Users/triplela0213/Developer/Projects/dissertation-cvr-llm python <script>.py
```

Without it, `huggingface_hub`'s default `hf_xet` transfer client hangs indefinitely at 0 bytes in this sandbox's network (confirmed 2026-08-16: a download sat at 0 bytes for ~1 hour before being killed). Plain HTTP download (`HF_HUB_DISABLE_XET=1`) works, just slowly (~500KB/s here) — budget real time for first-time model downloads, though they're cached afterward.

`sentence-transformers/all-mpnet-base-v2` (and likely other popular repos) has been migrated to HF's Xet storage: the `resolve/main/<file>` URL for both `pytorch_model.bin` and `model.safetensors` 302/308-redirects to a signed `us.aws.cdn.hf.co/xet-bridge-us/...` URL, not a plain LFS blob. Confirmed 2026-08-16: real throughput to that CDN from this sandbox is slow and highly variable (~6.8KB/s to ~58KB/s measured across repeated `curl` tests of the same file) — a 420MB model can take anywhere from ~2h to ~12h+. `HF_ENDPOINT=https://hf-mirror.com` does **not** help for this repo — it just 308-redirects straight back to `huggingface.co`'s own resolve URL (confirmed via `curl -sL -w '%{url_effective}'`), so any apparent speedup from trying it is just normal variance in the same underlying connection, not a faster route. Don't bother setting it.

If a download looks stalled, don't assume it's the `hf_xet` bug again — check whether the `.incomplete` blob in `~/.cache/huggingface/hub/models--.../blobs/` is actually growing (`ls -la` a few times, or watch its size in a loop) before deciding it's hung vs. just slow. If it's genuinely not growing at all (not just slow), it's probably the `hf_xet` bug — see above. If it's growing but very slowly, it's this Xet-bridge CDN throttling — either wait it out in the background, or (faster and more reliable) have the user download the needed model files via their own browser and load the model from a local folder path instead of the hub name.
