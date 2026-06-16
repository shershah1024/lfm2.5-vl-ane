import Foundation
import Lfm2VlKit

let args = CommandLine.arguments
let envAll = ProcessInfo.processInfo.environment

// Headless server mode: `lfm2vl-cli serve [port]` (bundle from $LFM2_BUNDLE).
// Runs LFM2.5-VL as a localhost /caption service — sibling to the LLM daemon.
if args.count >= 2 && args[1] == "serve" {
    let bundle = URL(fileURLWithPath: envAll["LFM2_BUNDLE"] ?? "bundle")
    let port = UInt16(args.count > 2 ? args[2] : (envAll["LFM2_PORT"] ?? "8766")) ?? 8766
    // Keep the async main task alive forever so the process (and the NWListener,
    // which serves on its own dispatch queues) stays up. `dispatchMain()` does NOT
    // block when called from an async top-level main — the task completes and the
    // process exits. Suspend on a sleep loop instead, holding `server` each pass.
    let server = try await VlServer.start(bundle: bundle, port: port)
    while true {
        try? await Task.sleep(nanoseconds: 3_600_000_000_000)
        _ = server   // referenced each iteration → ARC retains it for the process's life
    }
}

// usage: lfm2vl-cli <bundle> <image> [question]
guard args.count >= 3 else {
    print("usage: lfm2vl-cli <bundle-dir> <image> [question]\n       lfm2vl-cli serve [port]   (LFM2_BUNDLE env)"); exit(1)
}
let bundle = URL(fileURLWithPath: args[1])
let image = URL(fileURLWithPath: args[2])
let question = args.count > 3 ? args[3] : "What do you see in this image?"

let t0 = Date()
let engine = try await Lfm2Vl(bundle: bundle)
print("loaded in \(String(format: "%.1f", -t0.timeIntervalSinceNow))s")

// Optional decode overrides (e.g. for JSON/bbox output, where the default repetition
// penalty + no-repeat-ngram corrupt legitimately-repeating tokens):
//   LFM2_MAXTOK, LFM2_PENALTY, LFM2_NOREPEAT
let env = ProcessInfo.processInfo.environment
let maxTok = env["LFM2_MAXTOK"].flatMap { Int($0) } ?? 40
let penalty = env["LFM2_PENALTY"].flatMap { Float($0) } ?? 1.3
let noRep = env["LFM2_NOREPEAT"].flatMap { Int($0) } ?? 3

let t1 = Date()
let caption = try engine.caption(image: image, question: question,
                                 maxNewTokens: maxTok, repetitionPenalty: penalty, noRepeatNGram: noRep)
print("\nQ: \(question)")
print("A: \(caption)")
print("(\(String(format: "%.1f", -t1.timeIntervalSinceNow))s)")
