import Foundation

/// Decodes bundle/manifest.json — the full pipeline spec written by make_bundle.py.
public struct Manifest: Codable {
    public struct LayerRef: Codable { public let index: Int; public let name: String; public let type: String }
    public struct Language: Codable {
        public let hidden_size, num_layers, num_heads, num_kv_heads, head_dim, vocab_size, seq_len_T: Int
        public let norm_eps, rope_theta: Double
        public let layer_order: [LayerRef]
        public let conv_state_len: Int
        public let lang_dir: String
        public let decode_function: String
        public let prefill_function: String
    }
    public struct Vision: Codable {
        public let grid, patches, patch_size, tile_size, channels, image_tokens, downsample_factor: Int
        public let image_mean, image_std: [Float]
        public let vision_tower, projector: String
    }
    public struct Tokens: Codable {
        public let image, image_start, image_end, im_start, im_end, bos, eos, pad: Int
    }
    public struct Weights: Codable {
        public let embed_tied_lm_head: String
        public let embed_shape: [Int]
        public let embedding_norm: String
    }
    public let language: Language
    public let vision: Vision
    public let tokens: Tokens
    public let weights: Weights

    public static func load(_ bundleURL: URL) throws -> Manifest {
        let data = try Data(contentsOf: bundleURL.appendingPathComponent("manifest.json"))
        return try JSONDecoder().decode(Manifest.self, from: data)
    }
}
