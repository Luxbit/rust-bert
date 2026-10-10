// Copyright 2023 Laurent Mazare
// https://github.com/LaurentMazare/diffusers-rs/blob/main/build.rs
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

fn main() {
    // `doctest` is set by rustdoc; declare it explicitly so `cargo build`/
    // clippy do not warn about the `#[cfg(doctest)]` README doctest holder.
    println!("cargo:rustc-check-cfg=cfg(doctest)");

    // The libtorch link flags below are only required when the `tch` (LibTorch)
    // backend is enabled; ONNX-only builds must not link against libtorch.
    if std::env::var_os("CARGO_FEATURE_LIBTORCH").is_none() {
        return;
    }

    let os = std::env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    match os.as_str() {
        "linux" | "windows" => {
            if let Some(lib_path) = std::env::var_os("DEP_TCH_LIBTORCH_LIB") {
                println!(
                    "cargo:rustc-link-arg=-Wl,-rpath={}",
                    lib_path.to_string_lossy()
                );
            }
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            println!("cargo:rustc-link-arg=-Wl,--copy-dt-needed-entries");
            println!("cargo:rustc-link-arg=-ltorch");
        }
        _ => {}
    }
}
