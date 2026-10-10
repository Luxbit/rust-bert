//! # Ready-to-use NLP pipelines and Transformer-based models
//!
//! Rust-native state-of-the-art Natural Language Processing models and pipelines. Port of Hugging Face's [Transformers library](https://github.com/huggingface/transformers), using [tch-rs](https://github.com/LaurentMazare/tch-rs) or [onnxruntime bindings](https://github.com/pykeio/ort) and pre-processing from [rust-tokenizers](https://github.com/guillaume-be/rust-tokenizers). Supports multi-threaded tokenization and GPU inference.
//! This repository exposes the model base architecture, task-specific heads (see below) and [ready-to-use pipelines](https://github.com/guillaume-be/rust-bert#ready-to-use-pipelines). [Benchmarks](https://github.com/guillaume-be/rust-bert#benchmarks) are documented in the README.
//!
//! Get started with tasks including question answering, named entity recognition, translation, summarization, text generation, conversational agents and more in just a few lines of code:
//! ```no_run
//! use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};
//!
//! # fn main() -> anyhow::Result<()> {
//! let qa_model = QuestionAnsweringModel::new(Default::default())?;
//!
//! let question = String::from("Where does Amy live ?");
//! let context = String::from("Amy lives in Amsterdam");
//! let answers = qa_model.predict(&[QaInput { question, context }], 1, 32);
//! # Ok(())
//! # }
//! ```
//!
//! Output:
//! ```no_run
//! # use rust_bert::pipelines::question_answering::Answer;
//! # let output =
//! [Answer {
//!     score: 0.9976,
//!     start: 13,
//!     end: 21,
//!     answer: String::from("Amsterdam"),
//! }]
//! # ;
//! ```
//!
//! The tasks currently supported include:
//! - Translation
//! - Summarization
//! - Multi-turn dialogue
//! - Zero-shot classification
//! - Sentiment Analysis
//! - Named Entity Recognition
//! - Part of Speech tagging
//! - Question-Answering
//! - Language Generation
//! - Sentence Embeddings
//! - Masked Language Model
//! - Keywords extraction
//!
//! More information on these can be found in the [`pipelines` module](./pipelines/index.html)
//! - Transformer models base architectures with customized heads. These allow to load pre-trained models for customized inference in Rust
//!
//! <details>
//! <summary> <b> Click to expand to display the supported models/tasks matrix </b> </summary>
//!
//!| |**Sequence classification**|**Token classification**|**Question answering**|**Text Generation**|**Summarization**|**Translation**|**Masked LM**|**Sentence Embeddings**|
//!:-----:|:----:|:----:|:-----:|:----:|:-----:|:----:|:----:|:----:
//!DistilBERT|✅|✅|✅| | | |✅| ✅|
//!MobileBERT|✅|✅|✅| | | |✅| |
//!DeBERTa|✅|✅|✅| | | |✅| |
//!DeBERTa (v2)|✅|✅|✅| | | |✅| |
//!FNet|✅|✅|✅| | | |✅| |
//!BERT|✅|✅|✅| | | |✅| ✅|
//!RoBERTa|✅|✅|✅| | | |✅| ✅|
//!GPT| | | |✅ | | | |  |
//!GPT2| | | |✅ | | | |  |
//!GPT-Neo| | | |✅ | | | | |
//!GPT-J| | | |✅ | | | | |
//!BART|✅| | |✅ |✅| | | |
//!Marian| | | |  | |✅| |  |
//!MBart|✅| | |✅ | | | |  |
//!M2M100| | | |✅ | | | |  |
//!NLLB| | | |✅ | | | |  |
//!Electra | |✅| | | | |✅|  |
//!ALBERT |✅|✅|✅| | | |✅| ✅ |
//!T5 | | | |✅ |✅|✅| | ✅ |
//!LongT5 | | | |✅ |✅| | |  |
//!XLNet|✅|✅|✅|✅ | | |✅|  |
//!Reformer|✅| |✅|✅ | | |✅|  |
//!ProphetNet| | | |✅ |✅ | | |  |
//!Longformer|✅|✅|✅| | | |✅|  |
//!Pegasus| | | | |✅| | |  |
//! </details>
//!
//! # Getting started
//!
//! The crate supports two inference backends, selected via cargo features:
//! - `libtorch`: the complete set of models and pipelines, running on the C++ LibTorch API via the [tch](https://github.com/LaurentMazare/tch-rs) crate.
//! - `onnx`: inference on models exported to ONNX, running on the [onnxruntime](https://onnxruntime.ai) C++ library via the [ort](https://github.com/pykeio/ort) crate.
//!
//! No inference backend is enabled by default; at least one must be selected explicitly (both may be enabled for a dual-backend build). Every pipeline is available with the ONNX backend only, allowing to build and run this crate without any LibTorch dependency:
//! ```toml
//! [dependencies]
//! rust-bert = { version = "0.25.0", features = ["onnx"] }
//! ```
//! (`remote` and `default-tls` stay enabled by default for model downloads; use `default-features = false` to drop them and add `rustls-tls` instead of `default-tls` if desired.)
//! With this configuration the models must be provided as ONNX exports (see the [ONNX Runtime backend](#onnx-runtime-backend-optional-onnx-feature) section below); PyTorch weight files (`.pt`) require the `libtorch` feature.
//! The `onnx-cuda` feature (which implies `onnx`) enables the CUDA execution provider for onnxruntime.
//!
//! ## LibTorch installation (`libtorch` feature)
//!
//! With the `libtorch` feature, this library relies on the [tch](https://github.com/LaurentMazare/tch-rs) crate for bindings to the C++ Libtorch API.
//! The libtorch library can be downloaded either automatically or manually. The following provides a reference on how to set-up your environment
//! to use these bindings, please refer to the [tch](https://github.com/LaurentMazare/tch-rs) for detailed information or support.
//!
//! Furthermore, this library relies on a cache folder for downloading pre-trained models.
//! This cache location defaults to `~/.cache/.rustbert`, but can be changed by setting the `RUSTBERT_CACHE` environment variable. Note that the language models used by this library are in the order of the 100s of MBs to GBs.
//!
//! ### Manual installation (recommended)
//!
//! 1. Download `libtorch` from <https://pytorch.org/get-started/locally/>. This package requires `v2.4`: if this version is no longer available on the "get started" page,
//!    the file should be accessible by modifying the target link, for example `https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip` for a Linux version with CUDA12.
//! 2. Extract the library to a location of your choice
//! 3. Set the following environment variables
//! ##### Linux:
//! ```bash
//! export LIBTORCH=/path/to/libtorch
//! export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
//! ```
//!
//! ##### Windows
//! ```powershell
//! $Env:LIBTORCH = "X:\path\to\libtorch"
//! $Env:Path += ";X:\path\to\libtorch\lib"
//! ```
//!
//! ### Automatic installation
//!
//! Alternatively, you can let the `build` script automatically download the `libtorch` library for you. The `libtorch-download` feature flag needs to be enabled.
//! The CPU version of libtorch will be downloaded by default. To download a CUDA version, please set the environment variable `TORCH_CUDA_VERSION` to `cu124`.
//! Note that the libtorch library is large (order of several GBs for the CUDA-enabled version) and the first build may therefore take several minutes to complete.
//!
//! ## ONNX Runtime backend (optional `onnx` feature)
//!
//! The `onnx` feature runs inference on models exported to ONNX through the [ort](https://github.com/pykeio/ort) crate (2.0, requiring onnxruntime >= 1.17) with bindings to the onnxruntime C++ library.
//! The `onnx` feature can be used on its own to run every pipeline without LibTorch, or alongside the `libtorch` feature.
//!
//! ### Manual installation (recommended)
//!
//! 1. Download an onnxruntime release (>= 1.17) for your platform from the [release page](https://github.com/microsoft/onnxruntime/releases) of the onnxruntime project.
//! 2. Extract the library to a location of your choice.
//! 3. Enable the `onnx` feature and add an explicit `ort` dependency with the `load-dynamic` feature, matching the version used by `rust-bert`.
//! 4. Set the `ORT_DYLIB_PATH` environment variable to the location of the extracted onnxruntime library (`onnxruntime.dll`/`libonnxruntime.so`/`libonnxruntime.dylib` depending on the operating system).
//!
//! ### Automatic installation
//!
//! Alternatively, the `onnx` feature alone is sufficient: `rust-bert` enables ort's `download-binaries` feature, and prebuilt onnxruntime binaries are downloaded and linked automatically at build time.
//!
//! ONNX models are exported with the [Optimum](https://github.com/huggingface/optimum) library (see the
//! [export guide](https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model)).
//! All pipelines support ONNX checkpoints and reuse the PyTorch configuration and tokenizer files.
//! See the [`onnx` module](crate::pipelines::onnx) for the expected export layout
//! (encoder / decoder / decoder-with-past files) and a complete example.
//!
//! # Ready-to-use pipelines
//!
//! Ready-to-use end-to-end NLP pipelines (question answering, translation, summarization, text generation,
//! zero-shot classification, sentiment analysis, named-entity recognition, keywords extraction,
//! part-of-speech tagging, sentence embeddings, masked language modeling, ...) are documented, with runnable
//! examples, in the [README](https://github.com/guillaume-be/rust-bert#ready-to-use-pipelines).
//! More information on the individual pipelines can be found in the [`pipelines` module](./pipelines/index.html).
//!
//! **Disclaimer**
//! The contributors of this repository are not responsible for any generation from the 3rd party utilization of the pretrained systems proposed herein.
//!
//! ## Benchmarks
//!
//! See the [Benchmarks](https://github.com/guillaume-be/rust-bert#benchmarks) section of the README.
//!
//! ## Loading pretrained and custom model weights
//!
//! See [Loading pretrained and custom model weights](https://github.com/guillaume-be/rust-bert#loading-pretrained-and-custom-model-weights)
//! in the README for guidance on converting PyTorch checkpoints and loading custom weights.
//!
//! ## Async execution
//!
//! Creating any of the models in async context will cause panics! Running extensive calculations like running predictions in a future should be avoided, too ([see here](https://docs.rs/tokio/latest/tokio/#cpu-bound-tasks-and-blocking-code)).
//!
//! It is recommended to spawn a separate thread for the models. The `async-sentiment` example displays a possible solution you could use to integrate models into async code.
//!
//!
//! ## Citation
//!
//! See the [Citation](https://github.com/guillaume-be/rust-bert#citation) section of the README.
//!
//! ## Acknowledgements
//!
//! See the [Acknowledgements](https://github.com/guillaume-be/rust-bert#acknowledgements) section of the README.

// These are used abundantly in this code
#![allow(
    clippy::assign_op_pattern,
    clippy::upper_case_acronyms,
    clippy::unnecessary_unwrap
)]

extern crate core;

#[cfg(all(not(feature = "libtorch"), not(feature = "onnx")))]
compile_error!("rust-bert requires an inference backend: enable the `libtorch` or the `onnx` feature.");

#[cfg(all(feature = "remote", not(feature = "default-tls"), not(feature = "rustls-tls")))]
compile_error!("the `remote` feature requires a TLS backend: enable `default-tls` or `rustls-tls`.");

// Compile the examples in the README as doctests. This item only exists while
// rustdoc collects doctests, so the README is not duplicated in the rendered
// crate documentation while still being compiled by `cargo test --doc`.
#[cfg(all(doctest, feature = "libtorch", feature = "remote"))]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

mod common;
pub mod models;
pub mod pipelines;

pub use common::device::Device;
pub use common::error::RustBertError;
pub use common::resources;
#[cfg(feature = "libtorch")]
pub use common::Activation;
pub use common::Config;
pub use models::{
    albert, bert, deberta, deberta_v2, distilbert, electra, fnet, gpt2, longformer, m2m_100, mbart,
    mobilebert, xlnet,
};
#[cfg(feature = "libtorch")]
pub use models::{
    bart, gpt_j, gpt_neo, longt5, marian, nllb, openai_gpt, pegasus, prophetnet, reformer, roberta,
    t5,
};
