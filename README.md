# LFM2.5-VL on the Apple Neural Engine

A vision-language model ([LiquidAI **LFM2.5-VL-450M**](https://huggingface.co/LiquidAI/LFM2.5-VL-450M))
converted to run **entirely on the Apple Neural Engine** — image in, grounded answer out, at
~1–2 W, fully on-device, no cloud, no Python at runtime.

Ships with a **native macOS app** (drop an image, type a prompt, read the answer) and an
embedded **REST endpoint**, both built on a small Swift package (`Lfm2VlKit`) that loads the
model bundle and runs the whole pipeline on the ANE.

→ **[Why the Neural Engine — when to use it over the GPU](WHY_ANE.md)**

---

## Quick start

Requirements: **Apple Silicon Mac, macOS 15+, Swift 6 / Xcode 16.**

```bash
git clone <this repo> && cd lfm2-vl-ane
./scripts/run_app.sh          # fetches the ~500 MB bundle, builds, opens the app
```

- **GUI:** a window opens — choose/drop an image, edit the prompt, hit **Run** (⌘↵), read the response.
- **REST** (running inside the app, port 8765):
  ```bash
  IMG=$(base64 -i demo.png | tr -d '\n')
  curl -s localhost:8765/caption -d "{\"image\":\"$IMG\",\"prompt\":\"What do you see?\"}"
  # -> {"answer":"...","ms":1850}
  ```
- **CLI (one-shot):**
  ```bash
  ./scripts/run_cli.sh demo.png "Describe the elements in this image."
  ```
- **In your own app:**
  ```swift
  import Lfm2VlKit
  let engine = try await Lfm2Vl(bundle: bundleURL)          // .cpuAndNeuralEngine
  let answer = try engine.caption(image: imageURL, question: "Describe this image.")
  ```

The model **bundle** (compiled CoreML, ~500 MB) is hosted, not committed (it contains files
>100 MB). `run_app.sh` / `run_cli.sh` fetch it automatically; or run `./scripts/fetch_bundle.sh`.

---

## What's in the bundle

```
bundle/
  models/
    vision_tower.mlmodelc  projector.mlmodelc      SigLIP2, 512² tile → 256 image tokens (fp16)
    lang/  16 multifunction .mlmodelc              each holds a decode (seq=1) AND prefill (seq=S)
                                                   function that SHARE one 8-bit weight set
  weights/   tied embed/lm-head + final norm (fp16)
  tokenizer/ tokenizer.json + chat template + config
  manifest.json   full pipeline spec
```

## Performance (Apple Silicon, on-ANE)

| | |
|---|---|
| Prefill (one-pass, 275-token prompt incl. image) | **~50 ms** |
| Decode (KV cache, O(1)/token) | **~62 tok/s** |
| Full caption after load | **~0.8 s** |
| Power | **~1–2 W** (ANE) vs ~8–15 W on the GPU |
| Neural Engine residency | **100%** of language + vision ops |
| Bundle | ~590 MB on disk |

## What it's good at — and not

- **Great:** describing natural images and scenes, identifying objects, layout, colors, "what's in this."
  LFM2.5 also adds **visual grounding** (bounding-box prediction) and **multilingual** captions
  (en, zh, ja, ko, fr, es, de, ar, pt) — prompt-driven, no extra setup.
- **Weak:** transcribing **dense text / document pages**. The model is 450M params and uses a single
  512² image tile, so an A4 page gets squashed and body text becomes unreadable — it'll give you an
  accurate *gist* but it is **not an OCR/document reader**.
- **Tips:** prompt it to *describe the elements* rather than *transcribe the text*; the decoder uses a
  repetition penalty so it won't loop. See [WHY_ANE.md](WHY_ANE.md) for the full tradeoffs.

## About the model — Liquid AI

The model this project runs, **[LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M)**, is an
open-weight vision-language model from **[Liquid AI](https://www.liquid.ai/)**, part of their LFM2
family of on-device foundation models (the refreshed successor to LFM2-VL-450M, adding visual grounding
and multilingual support). This repository is an independent **CoreML / Apple Neural Engine port** of
that model — all model credit belongs to Liquid AI. See the license terms below.

## Model & license

- **Code** (the `Lfm2VlKit` package + `scripts/`): **MIT** — see [`LICENSE`](LICENSE).
- **Model bundle**: a Derivative Work of **[LiquidAI/LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M)**
  (CoreML conversion + 8-bit quantization), governed by the **LFM Open License v1.0** —
  see [`MODEL_LICENSE`](MODEL_LICENSE) and [`NOTICE`](NOTICE).

The LFM Open License permits redistribution and modified/derivative works (which is why this
bundle can be shared), with attribution. **Commercial use is licensed only for entities under
USD $10M annual revenue**; at or above that threshold a separate commercial license from Liquid AI
is required. See `MODEL_LICENSE` for the exact terms.
