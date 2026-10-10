# rust-bert

[![Build Status](https://github.com/guillaume-be/rust-bert/workflows/Build/badge.svg?event=push)](https://github.com/guillaume-be/rust-bert/actions)
[![Latest version](https://img.shields.io/crates/v/rust_bert.svg)](https://crates.io/crates/rust_bert)
[![Documentation](https://docs.rs/rust-bert/badge.svg)](https://docs.rs/rust-bert)
![License](https://img.shields.io/crates/l/rust_bert.svg)

Rust-native state-of-the-art Natural Language Processing models and pipelines.
Port of Hugging Face's
[Transformers library](https://github.com/huggingface/transformers), with
pre-processing from [rust-tokenizers](https://github.com/guillaume-be/rust-tokenizers).
Every pipeline runs on either of two interchangeable inference backends selected
at compile time: **LibTorch** (via [tch](https://github.com/LaurentMazare/tch-rs))
or **ONNX Runtime** (via [ort](https://github.com/pykeio/ort), with
no LibTorch dependency), both with multi-threaded tokenization and GPU
inference. See [Choose your inference backend](#choose-your-inference-backend)
for the trade-offs and dependency configurations. This repository exposes the
model base architecture, task-specific heads (see below) and
[ready-to-use pipelines](#ready-to-use-pipelines).
[Benchmarks](#benchmarks) are available at the end of this document.

Get started with tasks including question answering, named entity recognition,
translation, summarization, text generation, conversational agents and more in
just a few lines of code:

```rust,no_run
use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};

fn main() -> anyhow::Result<()> {
    let qa_model = QuestionAnsweringModel::new(Default::default())?;

let question = String::from("Where does Amy live ?");
let context = String::from("Amy lives in Amsterdam");

let answers = qa_model.predict(&[QaInput { question, context }], 1, 32);
    Ok(())
}
```

Output:

```text
[Answer { score: 0.9976, start: 13, end: 21, answer: "Amsterdam" }]
```

The tasks currently supported include:

- Translation
- Summarization
- Multi-turn dialogue
- Zero-shot classification
- Sentiment Analysis
- Named Entity Recognition
- Part of Speech tagging
- Question-Answering
- Language Generation
- Masked Language Model
- Sentence Embeddings
- Keywords extraction

<details>
<summary> <b>Expand to display the supported models/tasks matrix </b> </summary>

|              | **Sequence classification** | **Token classification** | **Question answering** | **Text Generation** | **Summarization** | **Translation** | **Masked LM** | **Sentence Embeddings** |
|:------------:|:---------------------------:|:------------------------:|:----------------------:|:-------------------:|:-----------------:|:---------------:|:-------------:|:-----------------------:|
|  DistilBERT  |              ✅              |            ✅             |           ✅            |                     |                   |                 |       ✅       |            ✅            |
|  MobileBERT  |              ✅              |            ✅             |           ✅            |                     |                   |                 |       ✅       |                         |
|   DeBERTa    |              ✅              |            ✅             |           ✅            |                     |                   |                 |       ✅       |                         |
| DeBERTa (v2) |              ✅              |            ✅             |           ✅            |                     |                   |                 |       ✅       |                         |
|     FNet     |              ✅              |            ✅             |           ✅            |                     |                   |                 |       ✅       |                         |
|     BERT     |              ✅              |            ✅             |           ✅            |                     |                   |                 |       ✅       |            ✅            |
|   RoBERTa    |              ✅              |            ✅             |           ✅            |                     |                   |                 |       ✅       |            ✅            |
|     GPT      |                             |                          |                        |          ✅          |                   |                 |               |                         |
|     GPT2     |                             |                          |                        |          ✅          |                   |                 |               |                         |
|   GPT-Neo    |                             |                          |                        |          ✅          |                   |                 |               |                         |
|    GPT-J     |                             |                          |                        |          ✅          |                   |                 |               |                         |
|     BART     |              ✅              |                          |                        |          ✅          |         ✅         |                 |               |                         |
|    Marian    |                             |                          |                        |                     |                   |        ✅        |               |                         |
|    MBart     |              ✅              |                          |                        |          ✅          |                   |                 |               |                         |
|    M2M100    |                             |                          |                        |          ✅          |                   |                 |               |                         |
|     NLLB     |                             |                          |                        |          ✅          |                   |                 |               |                         |
|   Electra    |                             |            ✅             |                        |                     |                   |                 |       ✅       |                         |
|    ALBERT    |              ✅              |            ✅             |           ✅            |                     |                   |                 |       ✅       |            ✅            |
|      T5      |                             |                          |                        |          ✅          |         ✅         |        ✅        |               |            ✅            |
|    LongT5    |                             |                          |                        |          ✅          |         ✅         |                 |               |                         |
|    XLNet     |              ✅              |            ✅             |           ✅            |          ✅          |                   |                 |       ✅       |                         |
|   Reformer   |              ✅              |                          |           ✅            |          ✅          |                   |                 |       ✅       |                         |
|  ProphetNet  |                             |                          |                        |          ✅          |         ✅         |                 |               |                         |
|  Longformer  |              ✅              |            ✅             |           ✅            |                     |                   |                 |       ✅       |                         |
|   Pegasus    |                             |                          |                        |                     |         ✅         |                 |               |                         |

</details>

&nbsp;

Every pipeline in the matrix runs on either backend; see
[Which backend should I pick?](#which-backend-should-i-pick) for the details.

## Getting started

### Choose your inference backend

The same pipelines and model APIs work with either backend; the difference is
which model files they load and which C++ library they link to. The backend is
selected with cargo features: **no backend is enabled by default**, so
`libtorch`, `onnx`, or both must be requested explicitly. The `remote` and
`default-tls` features stay on by default for model downloads.

| Backend | Cargo.toml |
|-------------|------------|
| LibTorch (tch), PyTorch `.pt` weights | `rust-bert = { version = "0.25.0", features = ["libtorch"] }` |
| ONNX Runtime, ONNX exports | `rust-bert = { version = "0.25.0", features = ["onnx"] }` |
| Both backends in the same binary | `rust-bert = { version = "0.25.0", features = ["libtorch", "onnx"] }` |

Notes:

- A backend must be selected explicitly (`libtorch`, `onnx`, or both); no
  backend is enabled by default, and enabling none is a compile error.
- The `remote` feature (enabled by default) lets pipelines download pretrained
  models from Hugging Face's hub; drop it if you only load local resources.
  Downloaded models are cached in `~/.cache/.rustbert` (override with the
  `RUSTBERT_CACHE` environment variable) and are in the order of 100s of MBs to
  GBs.
- `remote` requires a TLS backend: with `default-features = false`, add
  `default-tls` (the OS TLS stack) or `rustls-tls` (pure Rust) alongside it.
  Enabling `remote` without either is a compile error.
- `features = ["onnx-cuda"]` implies `onnx` and enables the onnxruntime CUDA
  execution provider. For LibTorch, GPU placement is selected through the
  device in the pipeline configuration (and requires a CUDA-enabled libtorch).

### Which backend should I pick?

For most applications the ONNX-only build is the better starting point: every
pipeline works on it and it avoids the LibTorch dependency entirely, with
models loaded as ONNX exports (ready-to-run examples in the `./examples`
directory: `onnx-question-answering`, `onnx-text-generation`,
`onnx-translation`, ...). The combined build (`libtorch` + `onnx`) is worth
enabling when any of the following applies:

- **Per-model backend choice.** The backend is selected per model instance
  (`ModelResource::Torch(...)` or `ModelResource::ONNX(...)` in the pipeline
  configuration), not per binary. With both features enabled, a single
  application can run some models from PyTorch weights and others from ONNX
  exports side by side.
- **Arbitrary PyTorch checkpoints.** The `libtorch` backend loads any converted
  `.pt` weights (see [Loading pretrained and custom model
  weights](#loading-pretrained-and-custom-model-weights)), while ONNX requires
  the model to have been exported first. Compatible exports are not available
  for every architecture, e.g. DialoGPT, XLNet, Reformer and ProphetNet have
  no widely available Optimum exports. The pipelines' built-in default
  resources also point at PyTorch checkpoints; ONNX checkpoints must be
  provided explicitly (hub URLs or local paths).
- **Torch-only capabilities.** `output_attentions` / `output_hidden_states`,
  custom heads built on top of the base models, and weight manipulation through
  the `VarStore` are LibTorch-only; ONNX is inference-only by nature.

The rest of this section covers the LibTorch installation; ONNX Runtime setup
is described in the [ONNX Runtime backend](#onnx-runtime-backend-optional-onnx-feature) section below.

### LibTorch installation (`libtorch` feature)

With the `libtorch` feature, this library relies on the
[tch](https://github.com/LaurentMazare/tch-rs) crate for bindings to the C++
Libtorch API; please refer to the
[tch](https://github.com/LaurentMazare/tch-rs) repository for detailed
information or support.

#### Manual installation (recommended)

1. Download `libtorch` from https://pytorch.org/get-started/locally/. This
   package requires `v2.4`: if this version is no longer available on the "get
   started" page, the file should be accessible by modifying the target link,
   for example
   `https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip`
   for a Linux version with CUDA12. **NOTE:** When using `rust-bert` as
   dependency from [crates.io](https://crates.io), please check the required
   `LIBTORCH` on the published package
   [readme](https://crates.io/crates/rust-bert) as it may differ from the
   version documented here (applying to the current repository version).
2. Extract the library to a location of your choice
3. Set the following environment variables

##### Linux:

```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
```

##### Windows

```powershell
$Env:LIBTORCH = "X:\path\to\libtorch"
$Env:Path += ";X:\path\to\libtorch\lib"
```

##### macOS + Homebrew

```bash
brew install pytorch jq
export LIBTORCH=$(brew --cellar pytorch)/$(brew info --json pytorch | jq -r '.[0].installed[0].version')
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
```

#### Automatic installation

Alternatively, you can let the `build` script automatically download the
`libtorch` library for you. The `libtorch-download` feature flag needs to be
enabled. The CPU version of libtorch will be downloaded by default. To download
a CUDA version, please set the environment variable `TORCH_CUDA_VERSION` to
`cu124`. Note that the libtorch library is large (order of several GBs for the
CUDA-enabled version) and the first build may therefore take several minutes to
complete.

#### Verifying installation

Verify your installation (and linking with libtorch) by adding the `rust-bert`
dependency to your `Cargo.toml` or by cloning the rust-bert source and running
an example:

```bash
git clone git@github.com:guillaume-be/rust-bert.git
cd rust-bert
cargo run --example sentence_embeddings
```

## ONNX Runtime backend (optional `onnx` feature)

The `onnx` feature runs inference on models exported to ONNX through the
[ort](https://github.com/pykeio/ort) crate (2.0, requiring onnxruntime >= 1.17)
with bindings to the onnxruntime C++ library; we refer the user to the ort
project page for further installation instructions/support.

### Manual installation (recommended)

1. Download an onnxruntime release (>= 1.17) for your platform from the
   [onnxruntime release page](https://github.com/microsoft/onnxruntime/releases),
   for example `onnxruntime-linux-x64-1.20.1.tgz`,
   `onnxruntime-osx-arm64-1.20.1.tgz` or `onnxruntime-win-x64-1.20.1.zip`.
2. Extract the library to a location of your choice.
3. Enable the `onnx` feature and add an explicit `ort` dependency with the
   `load-dynamic` feature, matching the version used by `rust-bert`:

   ```toml
   rust-bert = { version = "0.25.0", features = ["onnx"] }
   ort = { version = "=2.0.0-rc.13", default-features = false, features = ["load-dynamic"] }
   ```
4. Set the `ORT_DYLIB_PATH` environment variable to the location of the
   extracted shared library (`libonnxruntime.so` / `libonnxruntime.dylib` /
   `onnxruntime.dll` depending on the operating system):

##### Linux:

```bash
export ORT_DYLIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so
```

##### macOS:

```bash
export ORT_DYLIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.dylib
```

##### Windows

```powershell
$Env:ORT_DYLIB_PATH = "X:\path\to\onnxruntime\lib\onnxruntime.dll"
```

### Automatic installation

Alternatively, the `onnx` feature alone is sufficient: `rust-bert` enables
ort's `download-binaries` feature, and prebuilt onnxruntime binaries are
downloaded and linked automatically at build time. No environment variable is
required in this configuration, but the build will fetch the onnxruntime
library from the network on first compile.

### Verifying installation

Verify your installation (and linking with onnxruntime) by cloning the
rust-bert source and running an ONNX example:

```bash
git clone git@github.com:guillaume-be/rust-bert.git
cd rust-bert
export ORT_DYLIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so  # manual installation only
cargo run --features onnx --example onnx-question-answering
```

### Exporting models to ONNX

ONNX checkpoints are produced with
[Optimum](https://github.com/huggingface/optimum); see its
[export guide](https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model).
All pipelines support ONNX checkpoints and reuse the PyTorch configuration and
tokenizer files. Because ONNX graphs cannot handle optional arguments as
flexibly as PyTorch, decoder and encoder-decoder exports are split across up to
three files, passed to the pipelines through `ONNXModelResources`:

```rust,ignore
use rust_bert::pipelines::common::{ModelResource, ONNXModelResources};
use rust_bert::resources::RemoteResource;

let model_resource = ModelResource::ONNX(ONNXModelResources {
    encoder_resource: Some(Box::new(RemoteResource::new("encoder_model.onnx", "cache"))),
    decoder_resource: Some(Box::new(RemoteResource::new("decoder_model.onnx", "cache"))),
    decoder_with_past_resource: Some(Box::new(RemoteResource::new("decoder_with_past_model.onnx", "cache"))),
});
```

The required files depend on the architecture (encoder-only, decoder-only or
encoder-decoder) and the `decoder with past` file is optional; see the
[`onnx` module documentation](https://docs.rs/rust-bert/latest/rust_bert/pipelines/onnx/index.html)
for the file-layout table and a complete example.

## Ready-to-use pipelines

Based on Hugging Face's pipelines, ready to use end-to-end NLP pipelines are
available as part of this crate. The following capabilities are currently
available:

**Disclaimer** The contributors of this repository are not responsible for any
generation from the 3rd party utilization of the pretrained systems proposed
herein.

<details>
<summary> <b>1. Question Answering</b> </summary>

Extractive question answering from a given question and context. DistilBERT
model fine-tuned on SQuAD (Stanford Question Answering Dataset). See the
complete example at the top of this README.

</details>
&nbsp;
<details>
<summary> <b>2. Translation </b> </summary>

Translation pipeline supporting a broad range of source and target languages.
Leverages two main architectures for translation tasks:

- Marian-based models, for specific source/target combinations
- M2M100 models allowing for direct translation between 100 languages (at a
  higher computational cost and lower performance for some selected languages)

Marian-based pretrained models for the following language pairs are readily
available in the library - but the user can import any PyTorch-based model for
predictions

- English <-> French
- English <-> Spanish
- English <-> Portuguese
- English <-> Italian
- English <-> Catalan
- English <-> German
- English <-> Russian
- English <-> Chinese
- English <-> Dutch
- English <-> Swedish
- English <-> Arabic
- English <-> Hebrew
- English <-> Hindi
- French <-> German

For languages not supported by the proposed pretrained Marian models, the user
can leverage a M2M100 model supporting direct translation between 100 languages
(without intermediate English translation) The full list of supported languages
is available in the
[crate documentation](https://docs.rs/rust-bert/latest/rust_bert/pipelines/translation/enum.Language.html)

```rust,no_run
use rust_bert::pipelines::translation::{Language, TranslationModelBuilder};
fn main() -> anyhow::Result<()> {
    let model = TranslationModelBuilder::new()
        .with_source_languages(vec![Language::English])
        .with_target_languages(vec![Language::Spanish, Language::French, Language::Italian])
        .create_model()?;
    let input_text = "This is a sentence to be translated";
    let output = model.translate(&[input_text], None, Language::French)?;
    for sentence in output {
        println!("{}", sentence);
    }
    Ok(())
}
```

Output:

```text
Il s'agit d'une phrase à traduire
```

</details>
&nbsp;
<details>
<summary> <b>3. Summarization </b> </summary>

Abstractive summarization using a pretrained BART model.

```rust,no_run
use rust_bert::pipelines::generation_utils::LanguageGenerator;
use rust_bert::pipelines::summarization::SummarizationModel;

fn main() -> anyhow::Result<()> {
    let summarization_model = SummarizationModel::new(Default::default())?;

let input = ["In findings published Tuesday in Cornell University's arXiv by a team of scientists \
from the University of Montreal and a separate report published Wednesday in Nature Astronomy by a team \
from University College London (UCL), the presence of water vapour was confirmed in the atmosphere of K2-18b, \
a planet circling a star in the constellation Leo. This is the first such discovery in a planet in its star's \
habitable zone, not too hot and not too cold for liquid water to exist. The Montreal team, led by Björn Benneke, \
used data from the NASA's Hubble telescope to assess changes in the light coming from K2-18b's star as the planet \
passed between it and Earth. They found that certain wavelengths of light, which are usually absorbed by water, \
weakened when the planet was in the way, indicating not only does K2-18b have an atmosphere, but the atmosphere \
contains water in vapour form. The team from UCL then analyzed the Montreal team's data using their own software \
and confirmed their conclusion. This was not the first time scientists have found signs of water on an exoplanet, \
but previous discoveries were made on planets with high temperatures or other pronounced differences from Earth. \
\"This is the first potentially habitable planet where the temperature is right and where we now know there is water,\" \
said UCL astronomer Angelos Tsiaras. \"It's the best candidate for habitability right now.\" \"It's a good sign\", \
said Ryan Cloutier of the Harvard–Smithsonian Center for Astrophysics, who was not one of either study's authors. \
\"Overall,\" he continued, \"the presence of water in its atmosphere certainly improves the prospect of K2-18b being \
a potentially habitable planet, but further observations will be required to say for sure. \"
K2-18b was first identified in 2015 by the Kepler space telescope. It is about 110 light-years from Earth and larger \
but less dense. Its star, a red dwarf, is cooler than the Sun, but the planet's orbit is much closer, such that a year \
on K2-18b lasts 33 Earth days. According to The Guardian, astronomers were optimistic that NASA's James Webb space \
telescope, scheduled for launch in 2021, and the European Space Agency's 2028 ARIEL program, could reveal more \
about exoplanets like K2-18b."];

let output = summarization_model.summarize(&input)?;
    Ok(())
}
```

(example from:
[WikiNews](https://en.wikinews.org/wiki/Astronomers_find_water_vapour_in_atmosphere_of_exoplanet_K2-18b))

Output:

```text
"Scientists have found water vapour on K2-18b, a planet 110 light-years from Earth. 
This is the first such discovery in a planet in its star's habitable zone. 
The planet is not too hot and not too cold for liquid water to exist."
```

</details>
&nbsp;
<details>
<summary> <b>4. Dialogue Model </b> </summary>

Conversation model based on Microsoft's
[DialoGPT](https://github.com/microsoft/DialoGPT). This pipeline allows the
generation of single or multi-turn conversations between a human and a model.
The DialoGPT's page states that

> The human evaluation results indicate that the response generated from
> DialoGPT is comparable to human response quality under a single-turn
> conversation Turing test.
> ([DialoGPT repository](https://github.com/microsoft/DialoGPT))

The model uses a `ConversationManager` to keep track of active conversations and
generate responses to them.

```rust,no_run
use rust_bert::pipelines::conversation::{ConversationManager, ConversationModel};

fn main() -> anyhow::Result<()> {
    let conversation_model = ConversationModel::new(Default::default())?;
    let mut conversation_manager = ConversationManager::new();

    let _conversation_id =
        conversation_manager.create("Going to the movies tonight - any suggestions?");
    let output = conversation_model.generate_responses(&mut conversation_manager);
    println!("{output:?}");
    Ok(())
}
```

Example output:

```text
"The Big Lebowski."
```

</details>
&nbsp;
<details>
<summary> <b>5. Natural Language Generation </b> </summary>

Generate language based on a prompt. GPT2 and GPT available as base models.
Include techniques such as beam search, top-k and nucleus sampling, temperature
setting and repetition penalty. Supports batch generation of sentences from
several prompts. Sequences will be left-padded with the model's padding token if
present, the unknown token otherwise. This may impact the results, it is
recommended to submit prompts of similar length for best results

```rust,no_run
use rust_bert::gpt2::GPT2Generator;
use rust_bert::pipelines::generation_utils::GenerateOptions;

fn main() -> anyhow::Result<()> {
    let model = GPT2Generator::new(Default::default())?;

let input_context_1 = "The dog";
let input_context_2 = "The cat was";

let generate_options = GenerateOptions {
    max_length: Some(30),
    ..Default::default()
};

let output = model.generate(Some(&[input_context_1, input_context_2]), Some(generate_options))?;
    Ok(())
}
```

Example output:

```text
[
    "The dog's owners, however, did not want to be named. According to the lawsuit, the animal's owner, a 29-year"
    "The dog has always been part of the family. \"He was always going to be my dog and he was always looking out for me"
    "The dog has been able to stay in the home for more than three months now. \"It's a very good dog. She's"
    "The cat was discovered earlier this month in the home of a relative of the deceased. The cat\'s owner, who wished to remain anonymous,"
    "The cat was pulled from the street by two-year-old Jazmine.\"I didn't know what to do,\" she said"
    "The cat was attacked by two stray dogs and was taken to a hospital. Two other cats were also injured in the attack and are being treated."
]
```

</details>
&nbsp;
<details>
<summary> <b>6. Zero-shot classification </b> </summary>

Performs zero-shot classification on input sentences with provided labels using
a model fine-tuned for Natural Language Inference.

```rust,no_run
use rust_bert::pipelines::zero_shot_classification::ZeroShotClassificationModel;

fn main() -> anyhow::Result<()> {
    let sequence_classification_model = ZeroShotClassificationModel::new(Default::default())?;

let input_sentence = "Who are you voting for in 2020?";
let input_sequence_2 = "The prime minister has announced a stimulus package which was widely criticized by the opposition.";
let candidate_labels = &["politics", "public health", "economics", "sports"];

let output = sequence_classification_model.predict_multilabel(
&[input_sentence, input_sequence_2],
candidate_labels,
None,
128,
);
    Ok(())
}
```

Output:

```text
[
  [ Label { "politics", score: 0.972 }, Label { "public health", score: 0.032 }, Label {"economics", score: 0.006 }, Label {"sports", score: 0.004 } ],
  [ Label { "politics", score: 0.975 }, Label { "public health", score: 0.0818 }, Label {"economics", score: 0.852 }, Label {"sports", score: 0.001 } ],
]
```

</details>
&nbsp;
<details>
<summary> <b>7. Sentiment analysis </b> </summary>

Predicts the binary sentiment for a sentence. DistilBERT model fine-tuned on
SST-2.

```rust,no_run
use rust_bert::pipelines::sentiment::SentimentModel;

fn main() -> anyhow::Result<()> {
    let sentiment_classifier = SentimentModel::new(Default::default())?;

let input = [
"Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.",
"This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo...",
"If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.",
];

let output = sentiment_classifier.predict(& input);
    Ok(())
}
```

(Example courtesy of [IMDb](http://www.imdb.com))

Output:

```text
[
    Sentiment { polarity: Positive, score: 0.9981985493795946 },
    Sentiment { polarity: Negative, score: 0.9927982091903687 },
    Sentiment { polarity: Positive, score: 0.9997248985164333 }
]
```

</details>
&nbsp;
<details>
<summary> <b>8. Named Entity Recognition </b> </summary>

Extracts entities (Person, Location, Organization, Miscellaneous) from text.
BERT cased large model fine-tuned on CoNNL03, contributed by the
[MDZ Digital Library team at the Bavarian State Library](https://github.com/dbmdz).
Models are currently available for English, German, Spanish and Dutch.

```rust,no_run
use rust_bert::pipelines::ner::NERModel;

fn main() -> anyhow::Result<()> {
    let ner_model = NERModel::new( Default::default())?;

let input = [
"My name is Amy. I live in Paris.",
"Paris is a city in France."
];

let output = ner_model.predict(& input);
    Ok(())
}
```

Output:

```text
[
  [
    Entity { word: "Amy", score: 0.9986, label: "I-PER" }
    Entity { word: "Paris", score: 0.9985, label: "I-LOC" }
  ],
  [
    Entity { word: "Paris", score: 0.9988, label: "I-LOC" }
    Entity { word: "France", score: 0.9993, label: "I-LOC" }
  ]
]
```

</details>
&nbsp;
<details>
<summary> <b>9. Keywords/keyphrases extraction</b> </summary>

Extract keywords and keyphrases extractions from input documents

```rust,no_run
use rust_bert::pipelines::keywords_extraction::KeywordExtractionModel;

fn main() -> anyhow::Result<()> {
    let keyword_extraction_model = KeywordExtractionModel::new(Default::default())?;

    let input = "Rust is a multi-paradigm, general-purpose programming language. \
       Rust emphasizes performance, type safety, and concurrency. Rust enforces memory safety (that is, \
       that all references point to valid memory) without requiring the use of a garbage collector or \
       reference counting present in other memory-safe languages. To simultaneously enforce \
       memory safety and prevent concurrent data races, Rust's borrow checker tracks the object lifetime \
       and variable scope of all references in a program during compilation. Rust is popular for \
       systems programming but also offers high-level features including functional programming constructs.";

    let output = keyword_extraction_model.predict(&[input])?;
}
```

Output:

```text
"rust" - 0.50910604
"programming" - 0.35731024
"concurrency" - 0.33825397
"concurrent" - 0.31229728
"program" - 0.29115444
```

</details>
&nbsp;
<details>
<summary> <b>10. Part of Speech tagging </b> </summary>

Extracts Part of Speech tags (Noun, Verb, Adjective...) from text.

```rust,no_run
use rust_bert::pipelines::pos_tagging::POSModel;

fn main() -> anyhow::Result<()> {
    let pos_model = POSModel::new( Default::default())?;

let input = ["My name is Bob"];

let output = pos_model.predict(& input);
    Ok(())
}
```

Output:

```text
[
    Entity { word: "My", score: 0.1560, label: "PRP" }
    Entity { word: "name", score: 0.6565, label: "NN" }
    Entity { word: "is", score: 0.3697, label: "VBZ" }
    Entity { word: "Bob", score: 0.7460, label: "NNP" }
]
```

</details>
&nbsp;
<details>
<summary> <b>11. Sentence embeddings </b> </summary>

Generate sentence embeddings (vector representation). These can be used for
applications including dense information retrieval.

```rust,no_run
use rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType};

fn main() -> anyhow::Result<()> {
    let model = SentenceEmbeddingsBuilder::remote(
SentenceEmbeddingsModelType::AllMiniLmL12V2
).create_model()?;

let sentences = [
"this is an example sentence",
"each sentence is converted"
];

let output = model.encode(& sentences)?;
    Ok(())
}
```

Output:

```text
[
    [-0.000202666, 0.08148022, 0.03136178, 0.002920636 ...],
    [0.064757116, 0.048519745, -0.01786038, -0.0479775 ...]
]
```

</details>
&nbsp;
<details>
<summary> <b>12. Masked Language Model </b> </summary>

Predict masked words in input sentences.

```rust,no_run
use rust_bert::pipelines::masked_language::MaskedLanguageModel;

fn main() -> anyhow::Result<()> {
    let model = MaskedLanguageModel::new(Default::default())?;

let sentences = [
"Hello I am a <mask> student",
"Paris is the <mask> of France. It is <mask> in Europe.",
];

let output = model.predict(&sentences)?;
    Ok(())
}
```

Output:

```text
[
    [MaskedToken { text: "college", id: 2267, score: 8.091}],
    [
        MaskedToken { text: "capital", id: 3007, score: 16.7249}, 
        MaskedToken { text: "located", id: 2284, score: 9.0452}
    ]
]
```

</details>

## Benchmarks

For simple pipelines (sequence classification, tokens classification, question
answering) the performance between Python and Rust is expected to be comparable.
This is because the most expensive part of these pipeline is the language model
itself, sharing a common implementation in the Torch backend. The
[End-to-end NLP Pipelines in Rust](https://www.aclweb.org/anthology/2020.nlposs-1.4/)
provides a benchmarks section covering all pipelines.

For text generation tasks (summarization, translation, conversation, free text
generation), significant benefits can be expected (up to 2 to 4 times faster
processing depending on the input and application). The article
[Accelerating text generation with Rust](https://guillaume-be.github.io/2020-11-21/generation_benchmarks)
focuses on these text generation applications and provides more details on the
performance comparison to Python.

## Loading pretrained and custom model weights

The base model and task-specific heads are also available for users looking to
expose their own transformer based models. Examples on how to prepare the data
using a native tokenizers Rust library are available in `./examples` for BERT,
DistilBERT, RoBERTa, GPT, GPT2 and BART. Note that when importing models from
PyTorch, the convention for parameters naming needs to be aligned with the Rust
schema. Loading of the pre-trained weights will fail if any of the model
parameters weights cannot be found in the weight files. If this quality check is
to be skipped, an alternative method `load_partial` can be invoked from the
variables store.

Pretrained models are available on Hugging Face's
[model hub](https://huggingface.co/models?filter=rust) and can be loaded using
`RemoteResource` defined in this library.

A conversion utility script is included in `./utils` to convert PyTorch weights
to a set of weights compatible with this library. This script requires Python
and `torch` to be set-up, and can be used as follows:
`python ./utils/convert_model.py path/to/pytorch_model.bin` where
`path/to/pytorch_model.bin` is the location of the original PyTorch weights.

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python utils/convert_model.py path/to/pytorch_model.bin
```

## Citation

If you use `rust-bert` for your work, please cite
[End-to-end NLP Pipelines in Rust](https://www.aclweb.org/anthology/2020.nlposs-1.4/):

```bibtex
@inproceedings{becquin-2020-end,
    title = "End-to-end {NLP} Pipelines in Rust",
    author = "Becquin, Guillaume",
    booktitle = "Proceedings of Second Workshop for NLP Open Source Software (NLP-OSS)",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.nlposs-1.4",
    pages = "20--25",
}
```

## Acknowledgements

Thank you to [Hugging Face](https://huggingface.co) for hosting a set of weights
compatible with this Rust library. The list of ready-to-use pretrained models is
listed at
[https://huggingface.co/models?filter=rust](https://huggingface.co/models?filter=rust).
