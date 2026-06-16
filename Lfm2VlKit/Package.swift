// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Lfm2VlKit",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "Lfm2VlKit", targets: ["Lfm2VlKit"]),
        .executable(name: "lfm2vl-cli", targets: ["lfm2vl-cli"]),
        .executable(name: "lfm2vl-app", targets: ["lfm2vl-app"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.17"),
    ],
    targets: [
        .target(
            name: "Lfm2VlKit",
            dependencies: [.product(name: "Transformers", package: "swift-transformers")]
        ),
        .executableTarget(
            name: "lfm2vl-cli",
            dependencies: ["Lfm2VlKit"]
        ),
        .executableTarget(
            name: "lfm2vl-app",
            dependencies: ["Lfm2VlKit"]
        ),
    ],
    // Swift 5-style code (CoreML/Vision via serialized access) — build in the
    // Swift 5 language mode so 6.x toolchains don't reject it on strict-concurrency
    // Sendable checks (NWConnection handler, Lfm2Vl crossing actor boundaries).
    swiftLanguageModes: [.v5]
)
