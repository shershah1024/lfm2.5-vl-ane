import Foundation
import Network

/// Minimal HTTP/1.1 server (no deps) exposing the same caption pipeline as the UI.
///   POST /caption   body: {"image": "<base64 PNG/JPEG>", "prompt": "..."}  -> {"answer": "...", "ms": 123}
///   GET  /          -> short API description (JSON)
final class HTTPServer: @unchecked Sendable {
    private let service: CaptionService
    private let listener: NWListener

    init(service: CaptionService, port: UInt16) throws {
        self.service = service
        // Bind loopback only — the caption endpoint is unauthenticated and must not be reachable from the LAN.
        let params = NWParameters.tcp
        params.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: port)!)
        self.listener = try NWListener(using: params)
        listener.newConnectionHandler = { [weak self] conn in
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
                self.read(conn, buffer: buf)   // need more bytes
            }
        }
    }

    /// Returns a full HTTP response once the request is complete; nil if more bytes are needed.
    private func tryHandle(_ buf: Data) -> Data? {
        guard let sep = buf.range(of: Data("\r\n\r\n".utf8)) else { return nil }
        let header = String(data: buf[..<sep.lowerBound], encoding: .utf8) ?? ""
        let lines = header.split(separator: "\r\n", omittingEmptySubsequences: false)
        guard let reqLine = lines.first else { return nil }
        let parts = reqLine.split(separator: " ")
        guard parts.count >= 2 else { return http(400, "text/plain", Data("bad request".utf8)) }
        let method = String(parts[0]), path = String(parts[1])

        if method == "GET" {
            let info = #"{"endpoint":"POST /caption","body":{"image":"<base64 PNG/JPEG>","prompt":"..."},"returns":{"answer":"...","ms":0}}"#
            return http(200, "application/json", Data(info.utf8))
        }
        guard method == "POST", path == "/caption" else {
            return http(404, "text/plain", Data("not found".utf8))
        }
        // wait for the whole body
        let clen = lines.compactMap { l -> Int? in
            let p = l.lowercased(); guard p.hasPrefix("content-length:") else { return nil }
            return Int(p.dropFirst("content-length:".count).trimmingCharacters(in: .whitespaces))
        }.first ?? 0
        let bodyStart = sep.upperBound
        let have = buf.distance(from: bodyStart, to: buf.endIndex)
        if have < clen { return nil }
        let body = buf[bodyStart..<buf.index(bodyStart, offsetBy: clen)]
        guard let obj = try? JSONSerialization.jsonObject(with: Data(body)) as? [String: Any],
              let b64 = obj["image"] as? String, let img = Data(base64Encoded: b64) else {
            return http(400, "application/json", Data(#"{"error":"expected JSON {image:base64, prompt}"}"#.utf8))
        }
        let prompt = (obj["prompt"] as? String) ?? ""
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".img")
        try? img.write(to: tmp); defer { try? FileManager.default.removeItem(at: tmp) }
        let r = service.run(imageURL: tmp, prompt: prompt)
        let payload: [String: Any] = ["answer": r.answer, "ms": Int(r.ms.rounded())]
        let json = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
        return http(200, "application/json", json)
    }

    private func http(_ code: Int, _ ctype: String, _ body: Data) -> Data {
        let reason = [200: "OK", 400: "Bad Request", 404: "Not Found"][code] ?? "OK"
        var head = "HTTP/1.1 \(code) \(reason)\r\n"
        head += "Content-Type: \(ctype)\r\n"
        head += "Content-Length: \(body.count)\r\n"
        head += "Access-Control-Allow-Origin: *\r\n"
        head += "Connection: close\r\n\r\n"
        return Data(head.utf8) + body
    }
}
