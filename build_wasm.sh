export RUSTFLAGS=--cfg=web_sys_unstable_apis 
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