import SwiftUI
import AppKit
import Foundation
import Network
import UniformTypeIdentifiers
import Lfm2VlKit

// MARK: - Caption service (thread-safe; serializes access to the engine's caches)
final class CaptionService: @unchecked Sendable {
    let engine: Lfm2Vl
    private let queue = DispatchQueue(label: "lfm2vl.caption")     // serial
    init(_ engine: Lfm2Vl) { self.engine = engine }
    func run(imageURL: URL, prompt: String, maxNew: Int = 96) -> (answer: String, ms: Double) {
        queue.sync {
            let t = Date()
            let q = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
            let ans = (try? engine.caption(image: imageURL,
                        question: q.isEmpty ? "What do you see in this image?" : q,
                        maxNewTokens: maxNew)) ?? "(inference error)"
            return (ans, -t.timeIntervalSinceNow * 1000)
        }
    }
}

// MARK: - App state (UI)
@MainActor final class AppState: ObservableObject {
    @Published var status = "loading model…"
    @Published var modelReady = false
    @Published var busy = false
    @Published var image: NSImage?
    @Published var prompt = "What do you see in this image?"
    @Published var response = ""
    @Published var timing = ""
    @Published var serverInfo = ""
    var imageURL: URL?
    private var service: CaptionService?
    private var server: HTTPServer?

    func start(bundle: URL, port: UInt16) async {
        do {
            let engine = try await Lfm2Vl(bundle: bundle)
            let svc = CaptionService(engine)
            self.service = svc
            self.server = try HTTPServer(service: svc, port: port)
            self.modelReady = true
            self.status = "model ready"
            self.serverInfo = "REST: POST http://localhost:\(port)/caption  {image:<base64>, prompt:\"…\"}"
        } catch {
            self.status = "load failed: \(error)"
        }
    }

    func pickImage() {
        let p = NSOpenPanel()
        p.allowedContentTypes = [.image]; p.allowsMultipleSelection = false
        if p.runModal() == .OK, let url = p.url { setImage(url) }
    }
    func setImage(_ url: URL) {
        imageURL = url; image = NSImage(contentsOf: url); response = ""; timing = ""
    }
    func runCaption() {
        guard let svc = service, let url = imageURL, !busy else { return }
        busy = true; response = ""; timing = "running…"; let p = prompt
        Task.detached {
            let r = svc.run(imageURL: url, prompt: p)
            await MainActor.run { self.response = r.answer; self.timing = String(format: "%.0f ms", r.ms); self.busy = false }
        }
    }
}

// MARK: - View
struct ContentView: View {
    @EnvironmentObject var s: AppState
    var body: some View {
        HStack(spacing: 0) {
            // left: image
            VStack {
                ZStack {
                    RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .controlBackgroundColor))
                    if let img = s.image {
                        Image(nsImage: img).resizable().scaledToFit().padding(8)
                    } else {
                        VStack(spacing: 8) {
                            Image(systemName: "photo.on.rectangle.angled").font(.system(size: 42)).foregroundStyle(.secondary)
                            Text("Drop an image here, or choose one").foregroundStyle(.secondary)
                        }
                    }
                }
                .frame(minWidth: 340, minHeight: 340)
                .onDrop(of: [.fileURL], isTargeted: nil) { providers in
                    providers.first?.loadItem(forTypeIdentifier: UTType.fileURL.identifier) { item, _ in
                        if let d = item as? Data, let url = URL(dataRepresentation: d, relativeTo: nil) {
                            Task { @MainActor in s.setImage(url) }
                        }
                    }
                    return true
                }
                Button { s.pickImage() } label: { Label("Choose Image…", systemImage: "folder") }
            }.padding()

            Divider()

            // right: prompt + result
            VStack(alignment: .leading, spacing: 12) {
                Text("Prompt").font(.headline)
                TextEditor(text: $s.prompt).font(.body).frame(height: 64)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
                HStack {
                    Button { s.runCaption() } label: { Label("Run", systemImage: "play.fill") }
                        .keyboardShortcut(.return, modifiers: .command)
                        .disabled(!s.modelReady || s.busy || s.imageURL == nil)
                    if s.busy { ProgressView().controlSize(.small) }
                    Spacer()
                    Text(s.timing).font(.caption).foregroundStyle(.secondary)
                }
                Text("Response").font(.headline)
                ScrollView {
                    Text(s.response.isEmpty ? "—" : s.response)
                        .frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled)
                }
                .frame(maxHeight: .infinity)
                .padding(8)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
                Spacer()
                Text(s.status).font(.caption).foregroundStyle(s.modelReady ? .green : .orange)
                Text(s.serverInfo).font(.caption2).foregroundStyle(.secondary).textSelection(.enabled)
            }.padding().frame(minWidth: 360)
        }
        .frame(minWidth: 760, minHeight: 460)
    }
}

// MARK: - App entry
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ n: Notification) {
        NSApp.setActivationPolicy(.regular); NSApp.activate(ignoringOtherApps: true)
    }
}

@main struct Lfm2VlApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @StateObject private var state = AppState()
    private let bundle: URL
    private let port: UInt16
    init() {
        let env = ProcessInfo.processInfo.environment
        bundle = URL(fileURLWithPath: env["LFM2_BUNDLE"] ?? "bundle")
        port = UInt16(env["LFM2_PORT"] ?? "8765") ?? 8765
    }
    var body: some Scene {
        WindowGroup("LFM2-VL on ANE") {
            ContentView().environmentObject(state)
                .task { await state.start(bundle: bundle, port: port) }
        }
        .windowResizability(.contentSize)
    }
}
