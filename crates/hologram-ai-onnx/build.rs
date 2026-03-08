fn main() {
    prost_build::compile_protos(&["proto/onnx.proto"], &["proto/"])
        .expect("failed to compile onnx.proto");
}
