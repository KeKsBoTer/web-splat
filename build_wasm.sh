export RUSTFLAGS='--cfg getrandom_backend="wasm_js"'
cargo build \
    --no-default-features \
    --target wasm32-unknown-unknown \
    --lib \
    --features npz \
    --profile web-release \
&& wasm-bindgen \
    --out-dir public \
    --web target/wasm32-unknown-unknown/web-release/web_splats.wasm \
    --no-typescript     