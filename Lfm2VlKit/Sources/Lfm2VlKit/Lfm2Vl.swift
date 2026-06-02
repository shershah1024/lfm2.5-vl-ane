import Foundation
import CoreML
import Accelerate
import ImageIO
import CoreGraphics
import Tokenizers

/// On-device LFM2-VL runtime (KV-cache decode). Loads a bundle from make_bundle.py and runs
/// image+text -> caption on the Neural Engine. Attention KV-cache + short-conv state are carried
/// as host-side I/O tensors (seq=1 decode blocks); decode is O(1)/token.
public final class Lfm2Vl {
    let bundle: URL
    let m: Manifest
    let vision: MLModel
    let projector: MLModel
    let blocks: [(model: MLModel, attn: Bool)]    // seq=1 decode blocks
    let pblocks: [(model: MLModel, attn: Bool)]   // seq=S prefill blocks
    let embed: [Float]          // [V*H] fp32, tied embed + lm_head
    let enorm: [Float]          // [H]
    let tokenizer: Tokenizer
    var cosF: [Float] = [], sinF: [Float] = []
    // caches (per layer index)
    var kCache: [Int: [Float]] = [:], vCache: [Int: [Float]] = [:], convState: [Int: [Float]] = [:]

    public init(bundle: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine) async throws {
        self.bundle = bundle
        self.m = try Manifest.load(bundle)
        let cfg = MLModelConfiguration(); cfg.computeUnits = computeUnits
        func load(_ rel: String, function: String? = nil) throws -> MLModel {
            let c = MLModelConfiguration(); c.computeUnits = computeUnits
            if let f = function { c.functionName = f }
            return try MLModel(contentsOf: bundle.appendingPathComponent(rel), configuration: c)
        }
        self.vision = try load(m.vision.vision_tower)
        self.projector = try load(m.vision.projector)
        // multifunction language blocks: same .mlmodelc, two functions (decode + prefill), shared weights on disk
        let ldir = m.language.lang_dir, df = m.language.decode_function, pf = m.language.prefill_function
        self.blocks = try m.language.layer_order.map {
            (try load("\(ldir)/\($0.name).mlmodelc", function: df), $0.type == "attn")
        }
        self.pblocks = try m.language.layer_order.map {
            (try load("\(ldir)/\($0.name).mlmodelc", function: pf), $0.type == "attn")
        }
        let V = m.language.vocab_size, H = m.language.hidden_size
        self.embed = Self.loadF16(bundle.appendingPathComponent(m.weights.embed_tied_lm_head), count: V * H)
        self.enorm = Self.loadF16(bundle.appendingPathComponent(m.weights.embedding_norm), count: H)
        self.tokenizer = try await AutoTokenizer.from(modelFolder: bundle.appendingPathComponent("tokenizer"))
        buildRope()
    }

    func buildRope() {
        let T = m.language.seq_len_T, HD = m.language.head_dim, theta = m.language.rope_theta
        cosF = [Float](repeating: 0, count: T * HD); sinF = cosF
        for p in 0..<T { for i in 0..<(HD/2) {
            let f = Double(p) / pow(theta, Double(2*i)/Double(HD))
            let c = Float(Foundation.cos(f)), s = Float(Foundation.sin(f))
            cosF[p*HD+i] = c; cosF[p*HD+i+HD/2] = c; sinF[p*HD+i] = s; sinF[p*HD+i+HD/2] = s
        }}
    }

    static func loadF16(_ url: URL, count: Int) -> [Float] {
        let d = try! Data(contentsOf: url)
        return d.withUnsafeBytes { raw in let h = raw.bindMemory(to: Float16.self); return (0..<count).map { Float(h[$0]) } }
    }

    // MARK: CoreML helpers
    func array(_ shape: [Int], _ data: [Float]) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        data.withUnsafeBufferPointer { a.dataPointer.assumingMemoryBound(to: Float.self).update(from: $0.baseAddress!, count: data.count) }
        return a
    }
    func floats(_ a: MLMultiArray) -> [Float] {
        let n = a.count
        let shape = a.shape.map { $0.intValue }, strides = a.strides.map { $0.intValue }
        let f16 = a.dataType == .float16
        let p16 = a.dataPointer.assumingMemoryBound(to: Float16.self)
        let p32 = a.dataPointer.assumingMemoryBound(to: Float.self)
        // Fast path: C-contiguous (no padding) -> bulk read.
        var cs = 1, contiguous = true
        for d in stride(from: shape.count - 1, through: 0, by: -1) {
            if strides[d] != cs { contiguous = false; break }; cs *= shape[d]
        }
        if contiguous {
            if f16 { return (0..<n).map { Float(p16[$0]) } }
            return Array(UnsafeBufferPointer(start: p32, count: n))
        }
        // Slow path: CoreML padded a small trailing dim -> gather via strides.
        var out = [Float](repeating: 0, count: n)
        var coord = [Int](repeating: 0, count: shape.count)
        for i in 0..<n {
            var off = 0; for d in 0..<shape.count { off += coord[d] * strides[d] }
            out[i] = f16 ? Float(p16[off]) : p32[off]
            var d = shape.count - 1
            while d >= 0 { coord[d] += 1; if coord[d] < shape[d] { break }; coord[d] = 0; d -= 1 }
        }
        return out
    }
    func run(_ model: MLModel, _ inputs: [String: MLMultiArray]) throws -> MLFeatureProvider {
        try model.prediction(from: try MLDictionaryFeatureProvider(dictionary: inputs))
    }
    func out(_ fp: MLFeatureProvider, _ name: String) -> [Float] { floats(fp.featureValue(for: name)!.multiArrayValue!) }

    // MARK: image -> patches [NP*CH]
    func patches(_ imageURL: URL) throws -> [Float] {
        let G = m.vision.grid, TS = m.vision.tile_size, CH = m.vision.channels
        guard let src = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw Err.image }
        var rgba = [UInt8](repeating: 0, count: TS*TS*4)
        // Use the image's OWN RGB colorspace so draw() is a straight copy (no color management) —
        // matches PIL, which reads raw decoded samples and ignores ICC. Fall back to sRGB.
        let space = (cg.colorSpace?.model == .rgb ? cg.colorSpace! : CGColorSpace(name: CGColorSpace.sRGB)!)
        let ctx = CGContext(data: &rgba, width: TS, height: TS, bitsPerComponent: 8, bytesPerRow: TS*4,
                            space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: TS, height: TS))
        var pix = [Float](repeating: 0, count: G*G*CH)
        for gy in 0..<G { for gx in 0..<G { let p = gy*G + gx
            for py in 0..<16 { for px in 0..<16 {
                let base = ((gy*16+py)*TS + (gx*16+px))*4
                for c in 0..<3 { pix[p*CH + py*48 + px*3 + c] = (Float(rgba[base+c])/255.0 - 0.5)/0.5 }
            }}
        }}
        return pix
    }

    func imageEmbeds(_ imageURL: URL) throws -> [Float] {
        let NP = m.vision.patches, CH = m.vision.channels
        let vh = out(try run(vision, ["pixel_values": try array([1, NP, CH], try patches(imageURL))]), "vision_hidden")
        return out(try run(projector, ["vision_hidden": try array([1, NP, CH], vh)]), "text_embed")
    }

    // MARK: one decode step at sequence position `pos`
    func step(_ h0: [Float], _ pos: Int) throws -> [Float] {
        let H = m.language.hidden_size, HD = m.language.head_dim, NKV = m.language.num_kv_heads, T = m.language.seq_len_T
        var h = h0
        let cos1 = Array(cosF[pos*HD..<(pos+1)*HD]), sin1 = Array(sinF[pos*HD..<(pos+1)*HD])
        var mask = [Float](repeating: -30000.0, count: T); for i in 0..<pos { mask[i] = 0 }  // finite (ANE: -inf -> NaN)
        for (i, (model, attn)) in blocks.enumerated() {
            if attn {
                let fp = try run(model, ["hidden_states": try array([1,1,H], h),
                    "cos": try array([1,1,HD], cos1), "sin": try array([1,1,HD], sin1),
                    "k_cache": try array([1,NKV,T,HD], kCache[i]!), "v_cache": try array([1,NKV,T,HD], vCache[i]!),
                    "mask": try array([1,1,1,T], mask)])
                h = out(fp, "output")
                let kn = out(fp, "k_new"), vn = out(fp, "v_new")   // [NKV*1*HD]
                for head in 0..<NKV { for d in 0..<HD {
                    kCache[i]![head*T*HD + pos*HD + d] = kn[head*HD + d]
                    vCache[i]![head*T*HD + pos*HD + d] = vn[head*HD + d]
                }}
            } else {
                let fp = try run(model, ["hidden_states": try array([1,1,H], h),
                    "conv_state": try array([1, H, m.language.conv_state_len], convState[i]!)])
                h = out(fp, "output"); convState[i] = out(fp, "conv_state_out")
            }
        }
        return h
    }

    /// one-pass prefill: embeds [T*H] (right-padded), n=real length -> last hidden [H]; seeds caches.
    func prefill(_ embeds: [Float], _ n: Int) throws -> [Float] {
        let H = m.language.hidden_size, HD = m.language.head_dim, NKV = m.language.num_kv_heads, T = m.language.seq_len_T
        var causal = [Float](repeating: 0, count: T*T)
        for q in 0..<T { for k in (q+1)..<T { causal[q*T+k] = -30000.0 } }  // finite (ANE: -inf -> NaN)
        var h = embeds
        for (i, (model, attn)) in pblocks.enumerated() {
            if attn {
                let fp = try run(model, ["hidden_states": try array([1,T,H], h),
                    "cos": try array([1,T,HD], cosF), "sin": try array([1,T,HD], sinF),
                    "mask": try array([1,1,T,T], causal)])
                h = out(fp, "output")
                kCache[i] = out(fp, "k_cache"); vCache[i] = out(fp, "v_cache")   // [1,NKV,T,HD]
            } else {
                let fp = try run(model, ["hidden_states": try array([1,T,H], h)])
                h = out(fp, "output")
                let bx = out(fp, "bx")                          // [1,H,T], flat hh*T+t
                let L = m.language.conv_state_len
                var st = [Float](repeating: 0, count: H*L)
                for hh in 0..<H { for l in 0..<L { st[hh*L+l] = bx[hh*T + (n-L+l)] } }
                convState[i] = st
            }
        }
        return Array(h[((n-1)*H)..<(n*H)])
    }

    func logits(_ h: [Float]) -> [Float] {
        let H = m.language.hidden_size, V = m.language.vocab_size
        var x = h
        var ss: Float = 0; vDSP_svesq(x, 1, &ss, vDSP_Length(H)); ss = ss/Float(H) + Float(m.language.norm_eps)
        var inv = 1.0/sqrt(ss); vDSP_vsmul(x, 1, &inv, &x, 1, vDSP_Length(H)); vDSP_vmul(x, 1, enorm, 1, &x, 1, vDSP_Length(H))
        var o = [Float](repeating: 0, count: V)
        embed.withUnsafeBufferPointer { e in
            cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(V), Int32(H), 1.0, e.baseAddress!, Int32(H), x, 1, 0.0, &o, 1)
        }
        return o
    }
    func argmax(_ a: [Float]) -> Int { var idx: vDSP_Length = 0; var mx: Float = 0; vDSP_maxvi(a, 1, &mx, &idx, vDSP_Length(a.count)); return Int(idx) }

    /// Greedy pick with a repetition penalty over already-generated tokens and a no-repeat-n-gram
    /// block — kills the "white car / blue sky / white car…" loops without changing the model.
    func pickNext(_ logits: [Float], generated: [Int], penalty: Float, noRepeatN: Int) -> Int {
        var l = logits
        if penalty != 1 {
            for t in Set(generated) where t < l.count {
                l[t] = l[t] > 0 ? l[t] / penalty : l[t] * penalty
            }
        }
        if noRepeatN > 1, generated.count >= noRepeatN - 1 {
            let k = noRepeatN - 1
            let prefix = Array(generated.suffix(k))
            var i = 0
            while i + k < generated.count {     // ban tokens that would repeat an existing n-gram
                if Array(generated[i..<i+k]) == prefix { l[generated[i+k]] = -.infinity }
                i += 1
            }
        }
        return argmax(l)
    }

    func resetCaches() {
        let H = m.language.hidden_size, HD = m.language.head_dim, NKV = m.language.num_kv_heads, T = m.language.seq_len_T
        kCache.removeAll(); vCache.removeAll(); convState.removeAll()
        for (i, o) in m.language.layer_order.enumerated() {
            if o.type == "attn" { kCache[i] = [Float](repeating: 0, count: NKV*T*HD); vCache[i] = kCache[i] }
            else { convState[i] = [Float](repeating: 0, count: H*m.language.conv_state_len) }
        }
    }

    /// image + question -> caption (greedy, KV cache).
    public func caption(image: URL, question: String = "What do you see in this image?",
                        maxNewTokens: Int = 40,
                        repetitionPenalty: Float = 1.3, noRepeatNGram: Int = 3) throws -> String {
        let H = m.language.hidden_size, T = m.language.seq_len_T, nImg = m.vision.image_tokens
        let imgEmb = try imageEmbeds(image)
        // NB: swift-transformers auto-prepends BOS (<|startoftext|>=1), so we omit it here.
        let prompt = "<|im_start|>user\n<|image_start|>"
            + String(repeating: "<image>", count: nImg)
            + "<|image_end|>" + question + "<|im_end|>\n<|im_start|>assistant\n"
        let ids = tokenizer.encode(text: prompt)
        let S = ids.count
        precondition(S <= T, "prompt \(S) exceeds T=\(T)")
        resetCaches()
        func embRow(_ t: Int) -> [Float] { Array(embed[(t*H)..<((t+1)*H)]) }
        // build right-padded [T*H] embeds with image-token scatter
        var embSeq = [Float](repeating: 0, count: T*H)
        var imgC = 0
        for (p, t) in ids.enumerated() {
            let e = (t == m.tokens.image) ? Array(imgEmb[(imgC*H)..<((imgC+1)*H)]) : embRow(t)
            if t == m.tokens.image { imgC += 1 }
            for j in 0..<H { embSeq[p*H+j] = e[j] }
        }
        // one-pass prefill seeds all caches
        let tp = Date()
        let lastH = try prefill(embSeq, S)
        let prefillMs = -tp.timeIntervalSinceNow * 1000
        // decode
        let td = Date()
        var gen: [Int] = [], pos = S
        var next = pickNext(logits(lastH), generated: gen, penalty: repetitionPenalty, noRepeatN: noRepeatNGram)
        while gen.count < maxNewTokens && next != m.tokens.eos && pos < T {
            gen.append(next)
            let h = try step(embRow(next), pos); pos += 1
            next = pickNext(logits(h), generated: gen, penalty: repetitionPenalty, noRepeatN: noRepeatNGram)
        }
        let dt = -td.timeIntervalSinceNow
        FileHandle.standardError.write(String(format: "prefill %.0f ms (S=%d) · decode %.1f tok/s (%d tokens)\n", prefillMs, S, Double(gen.count)/dt, gen.count).data(using: .utf8)!)
        return tokenizer.decode(tokens: gen)
    }

    enum Err: Error { case image }
}
