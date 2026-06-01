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

let t1 = Date()
let caption = try engine.caption(image: image, question: question)
print("\nQ: \(question)")
print("A: \(caption)")
print("(\(String(format: "%.1f", -t1.timeIntervalSinceNow))s)")
