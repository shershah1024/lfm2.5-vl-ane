# Why the Neural Engine

Both the GPU and the Neural Engine can run this model on-device. The GPU is actually **faster**
(~100 tok/s vs ~60–70 on the ANE) — but raw speed isn't the deciding factor; both are well past
interactive. What decides it is **how you're using it**: power, heat, and whether the GPU is free.
The GPU is the obvious choice for a one-off; the moment the model is embedded in something, or
sharing the machine with other GPU work, the ANE wins — you trade a bit of speed for ~1–2 W and a
free GPU.

## When the GPU, when the ANE

| Your situation | Run on | Why |
|---|---|---|
| One-off caption — point it at an image, wait a beat, read the answer | **GPU** | Fastest (~100 tok/s); a few seconds at ~10 W costs nothing, and nothing else is competing for the GPU. |
| Batch a folder of images, plugged in, machine otherwise idle | **GPU** | Highest throughput for a short job; power and heat don't matter when it's over in a minute. |
| **The GPU is already busy** — embedded in a tool you keep open, e.g. a terminal agent doing its *own* image descriptions locally (instead of calling Claude / Codex) on a stack that already runs Metal | **ANE** | The VLM rides the otherwise-idle Neural Engine at ~1–2 W, so the GPU stays free for the rest of your stack. ← *this project's case* |
| Always-on / background — watching the screen or camera continuously | **ANE** | ~1–2 W and cool, so it can run for hours without draining the battery or spinning the fans. |
| Alongside another Metal workload — a GPU-decoding LLM, a game, a video call, Blender | **ANE** | Runs on a separate engine, so it doesn't contend with or stall whatever's already on the GPU. |
| On battery, unplugged, or thermals matter | **ANE** | A fraction of the power; the GPU path runs hot and eats battery. |

**Rule of thumb:** occasional and in the foreground → GPU. Continuous, embedded, or sharing the
machine with other GPU work → ANE. This project is built for the second case: **100% of the vision
and language ops run on the Neural Engine**, at ~1–2 W vs the ~8–15 W the same model pulls on Metal.
