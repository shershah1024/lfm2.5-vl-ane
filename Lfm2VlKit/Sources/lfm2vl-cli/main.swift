import Foundation
import Lfm2VlKit

// usage: lfm2vl-cli <bundle> <image> [question]
let args = CommandLine.arguments
guard args.count >= 3 else {
    print("usage: lfm2vl-cli <bundle-dir> <image> [question]"); exit(1)
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
