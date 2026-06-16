import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Headless localhost caption/grounding server — the same `/caption` contract the
/// SwiftUI app exposes, but with no window, so LFM2.5-VL can run as a background
/// service (sibling to the LLM daemon) that other apps borrow over HTTP.
///
///   lfm2vl-cli serve [port]        (bundle from $LFM2_BUNDLE, default ./bundle)
///   curl localhost:8766/caption -d '{"image":"<base64>","prompt":"What do you see?"}'
///   → {"answer":"...","ms":812}
///
/// Loopback-only. One engine, serialized — KV-cache decode is single-threaded.
/// Uses raw BSD sockets (the proven agentmlx/AgentCore.Server pattern); an earlier
/// NWListener version accepted connections but never delivered a response on this
/// CLI runtime.
public final class VlServer: @unchecked Sendable {
    private let engine: Lfm2Vl

    public init(engine: Lfm2Vl) { self.engine = engine }

    /// Load the engine, then run the blocking accept loop on a background thread and
    /// return. The caller keeps the process alive (the loop runs until exit).
    public static func start(bundle: URL, port: UInt16) async throws -> VlServer {
        let t0 = Date()
        let engine = try await Lfm2Vl(bundle: bundle)
        FileHandle.standardError.write("[lfm2vl] loaded in \(String(format: "%.1f", -t0.timeIntervalSinceNow))s\n".data(using: .utf8)!)
        let server = VlServer(engine: engine)
        let thread = Thread { server.serve(port: port) }
        thread.stackSize = 4 << 20
        thread.start()
        return server
    }

    // MARK: raw-socket accept loop (single-threaded; the model is single-threaded)

    private func serve(port: UInt16) {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { perror("socket"); return }
        var yes: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")   // loopback only
        let ok = withUnsafePointer(to: &addr) { p in
            p.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard ok == 0 else { perror("bind"); return }
        listen(fd, 16)
        FileHandle.standardError.write("[lfm2vl] serving on http://127.0.0.1:\(port)  (POST /caption)\n".data(using: .utf8)!)
        while true {
            let client = accept(fd, nil, nil)
            if client < 0 { continue }
            if let req = readRequest(client) {
                let resp = handle(method: req.method, path: req.path, body: req.body)
                resp.withUnsafeBytes { _ = write(client, $0.baseAddress, resp.count) }
            }
            close(client)
        }
    }

    private func readRequest(_ client: Int32) -> (method: String, path: String, body: String)? {
        var data = Data()
        var buf = [UInt8](repeating: 0, count: 65536)
        while true {
            let n = read(client, &buf, buf.count)
            if n <= 0 { return nil }
            data.append(contentsOf: buf[0..<n])
            guard let sep = data.range(of: Data("\r\n\r\n".utf8)) else { continue }
            let header = String(data: data.subdata(in: 0..<sep.lowerBound), encoding: .utf8) ?? ""
            let lines = header.components(separatedBy: "\r\n")
            let first = lines.first?.split(separator: " ") ?? []
            let method = first.count > 0 ? String(first[0]) : ""
            let path = first.count > 1 ? String(first[1]) : "/"
            var contentLength = 0
            for l in lines where l.lowercased().hasPrefix("content-length:") {
                contentLength = Int(l.dropFirst("content-length:".count).trimmingCharacters(in: .whitespaces)) ?? 0
            }
            var body = data.subdata(in: sep.upperBound..<data.count)
            while body.count < contentLength {
                let n2 = read(client, &buf, buf.count)
                if n2 <= 0 { break }
                body.append(contentsOf: buf[0..<n2])
            }
            return (method, path, String(data: body, encoding: .utf8) ?? "")
        }
    }

    private func handle(method: String, path: String, body: String) -> Data {
        if method == "GET" {
            return http(200, "application/json",
                        #"{"endpoint":"POST /caption","body":{"image":"<base64>","prompt":"..."}}"#)
        }
        guard method == "POST", path.hasPrefix("/caption") else { return http(404, "text/plain", "not found") }
        guard let obj = (try? JSONSerialization.jsonObject(with: Data(body.utf8))) as? [String: Any],
              let b64 = obj["image"] as? String, let img = Data(base64Encoded: b64) else {
            return http(400, "application/json", #"{"error":"expected {image:base64, prompt}"}"#)
        }
        let prompt = (obj["prompt"] as? String) ?? ""
        let t = Date()
        let q = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let lc = q.lowercased()
        // Auto-relax decoding for structured/grounding prompts (rep-penalty + no-repeat
        // corrupt legitimately-repeating coordinate JSON), unless overridden.
        let structured = lc.contains("json") || lc.contains("bbox")
            || lc.contains("detect ") || lc.contains("[x1") || lc.contains("bounding")
        let mt = (obj["max_tokens"] as? Int) ?? (structured ? 224 : 96)
        let pen = (obj["penalty"] as? NSNumber)?.floatValue ?? (structured ? 1.0 : 1.3)
        let nr = (obj["no_repeat"] as? Int) ?? (structured ? 0 : 3)
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".img")
        try? img.write(to: tmp); defer { try? FileManager.default.removeItem(at: tmp) }
        let ans = (try? engine.caption(image: tmp,
                    question: q.isEmpty ? "What do you see in this image?" : q,
                    maxNewTokens: mt, repetitionPenalty: pen, noRepeatNGram: nr)) ?? "(inference error)"
        let payload: [String: Any] = ["answer": ans, "ms": Int((-t.timeIntervalSinceNow * 1000).rounded())]
        let json = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
        return http(200, "application/json", json)
    }

    private func http(_ code: Int, _ ctype: String, _ bodyString: String) -> Data {
        http(code, ctype, Data(bodyString.utf8))
    }
    private func http(_ code: Int, _ ctype: String, _ body: Data) -> Data {
        let reason = [200: "OK", 400: "Bad Request", 404: "Not Found"][code] ?? "OK"
        var head = "HTTP/1.1 \(code) \(reason)\r\n"
        head += "Content-Type: \(ctype)\r\nContent-Length: \(body.count)\r\n"
        head += "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
        return Data(head.utf8) + body
    }
}
