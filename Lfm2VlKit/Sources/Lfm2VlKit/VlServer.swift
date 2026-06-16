import Foundation
import Network

/// Headless localhost caption/grounding server — the same `/caption` contract the
/// SwiftUI app exposes, but with no window, so LFM2.5-VL can run as a background
/// service (sibling to the LLM daemon) that other apps borrow over HTTP.
///
///   lfm2vl-cli serve [port]        (bundle from $LFM2_BUNDLE, default ./bundle)
///   curl localhost:8766/caption -d '{"image":"<base64>","prompt":"What do you see?"}'
///   → {"answer":"...","ms":812}
///
/// Loopback-only: the endpoint is unauthenticated and must not be LAN-reachable.
/// One engine, serialized — the KV-cache decode is single-threaded.
public final class VlServer: @unchecked Sendable {
    private let engine: Lfm2Vl
    private let queue = DispatchQueue(label: "lfm2vl.serve")   // serialize cache access
    private var listener: NWListener!

    public init(engine: Lfm2Vl) { self.engine = engine }

    /// Load the engine and start the listener; returns the live server. The caller
    /// blocks the process (e.g. `dispatchMain()`) — listener callbacks run on
    /// global queues. `bundle` overridable via $LFM2_BUNDLE.
    public static func start(bundle: URL, port: UInt16) async throws -> VlServer {
        let t0 = Date()
        let engine = try await Lfm2Vl(bundle: bundle)
        FileHandle.standardError.write("[lfm2vl] loaded in \(String(format: "%.1f", -t0.timeIntervalSinceNow))s\n".data(using: .utf8)!)
        let server = VlServer(engine: engine)
        try server.start(port: port)
        FileHandle.standardError.write("[lfm2vl] serving on http://127.0.0.1:\(port)  (POST /caption)\n".data(using: .utf8)!)
        return server
    }

    private func start(port: UInt16) throws {
        let params = NWParameters.tcp
        params.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: port)!)
        listener = try NWListener(using: params)
        listener.newConnectionHandler = { @Sendable [weak self] conn in
            conn.start(queue: .global())
            self?.read(conn, buffer: Data())
        }
        listener.start(queue: .global())
    }

    private func read(_ conn: NWConnection, buffer: Data) {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, done, err in
            guard let self else { return }
            var buf = buffer
            if let d = data { buf.append(d) }
            if let resp = self.tryHandle(buf) {
                conn.send(content: resp, completion: .contentProcessed { _ in conn.cancel() })
            } else if done || err != nil {
                conn.cancel()
            } else {
                self.read(conn, buffer: buf)
            }
        }
    }

    private func tryHandle(_ buf: Data) -> Data? {
        guard let sep = buf.range(of: Data("\r\n\r\n".utf8)) else { return nil }
        let header = String(data: buf[..<sep.lowerBound], encoding: .utf8) ?? ""
        let lines = header.split(separator: "\r\n", omittingEmptySubsequences: false)
        guard let reqLine = lines.first else { return nil }
        let parts = reqLine.split(separator: " ")
        guard parts.count >= 2 else { return http(400, "text/plain", Data("bad request".utf8)) }
        let method = String(parts[0]), path = String(parts[1])

        if method == "GET" {
            let info = #"{"endpoint":"POST /caption","body":{"image":"<base64>","prompt":"..."}}"#
            return http(200, "application/json", Data(info.utf8))
        }
        guard method == "POST", path == "/caption" else {
            return http(404, "text/plain", Data("not found".utf8))
        }
        let clen = lines.compactMap { l -> Int? in
            let p = l.lowercased(); guard p.hasPrefix("content-length:") else { return nil }
            return Int(p.dropFirst("content-length:".count).trimmingCharacters(in: .whitespaces))
        }.first ?? 0
        let bodyStart = sep.upperBound
        if buf.distance(from: bodyStart, to: buf.endIndex) < clen { return nil }
        let body = buf[bodyStart..<buf.index(bodyStart, offsetBy: clen)]
        guard let obj = try? JSONSerialization.jsonObject(with: Data(body)) as? [String: Any],
              let b64 = obj["image"] as? String, let img = Data(base64Encoded: b64) else {
            return http(400, "application/json", Data(#"{"error":"expected {image:base64, prompt}"}"#.utf8))
        }
        let prompt = (obj["prompt"] as? String) ?? ""
        let r = run(image: img, prompt: prompt,
                    maxNew: obj["max_tokens"] as? Int,
                    penalty: (obj["penalty"] as? NSNumber)?.floatValue,
                    noRepeat: obj["no_repeat"] as? Int)
        let payload: [String: Any] = ["answer": r.answer, "ms": Int(r.ms.rounded())]
        let json = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
        return http(200, "application/json", json)
    }

    /// Serialized inference. Auto-relaxes decoding for structured/grounding prompts
    /// (the caption rep-penalty + no-repeat-ngram corrupt legitimately-repeating
    /// coordinate JSON), unless the caller overrides.
    private func run(image: Data, prompt: String, maxNew: Int?, penalty: Float?, noRepeat: Int?) -> (answer: String, ms: Double) {
        queue.sync {
            let t = Date()
            let q = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
            let lc = q.lowercased()
            let structured = lc.contains("json") || lc.contains("bbox")
                || lc.contains("detect ") || lc.contains("[x1") || lc.contains("bounding")
            let mt = maxNew ?? (structured ? 224 : 96)
            let pen = penalty ?? (structured ? 1.0 : 1.3)
            let nr = noRepeat ?? (structured ? 0 : 3)
            let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".img")
            try? image.write(to: tmp); defer { try? FileManager.default.removeItem(at: tmp) }
            let ans = (try? engine.caption(image: tmp,
                        question: q.isEmpty ? "What do you see in this image?" : q,
                        maxNewTokens: mt, repetitionPenalty: pen, noRepeatNGram: nr)) ?? "(inference error)"
            return (ans, -t.timeIntervalSinceNow * 1000)
        }
    }

    private func http(_ code: Int, _ ctype: String, _ body: Data) -> Data {
        let reason = [200: "OK", 400: "Bad Request", 404: "Not Found"][code] ?? "OK"
        var head = "HTTP/1.1 \(code) \(reason)\r\n"
        head += "Content-Type: \(ctype)\r\nContent-Length: \(body.count)\r\n"
        head += "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
        return Data(head.utf8) + body
    }
}
