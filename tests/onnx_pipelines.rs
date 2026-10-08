#![cfg(feature = "onnx")]
//! Coverage tests for the ONNX-only pipeline paths: sentence embeddings, keywords
//! extraction, conversation, zero-shot classification, summarization and the
//! decoder-without-past generation fallback.
//!
//! All remote ONNX resources pin a commit revision so the asserted golden values
//! cannot drift when the Hugging Face repositories are updated.

mod tests {
    extern crate anyhow;

    use rust_bert::bert::BertConfigResources;
    use rust_bert::pipelines::common::TokenizerOption;
    use rust_bert::pipelines::common::{ModelResource, ModelType, ONNXModelResources};
    use rust_bert::pipelines::conversation::{
        ConversationConfig, ConversationManager, ConversationModel,
    };
    use rust_bert::pipelines::keywords_extraction::{
        KeywordExtractionConfig, KeywordExtractionModel, KeywordScorerType,
    };
    use rust_bert::pipelines::sentence_embeddings::{
        SentenceEmbeddingsConfig, SentenceEmbeddingsModel,
    };
    use rust_bert::pipelines::sentence_embeddings::{
        SentenceEmbeddingsConfigResources, SentenceEmbeddingsModulesConfigResources,
        SentenceEmbeddingsPoolingConfigResources, SentenceEmbeddingsTokenizerConfigResources,
    };
    use rust_bert::pipelines::summarization::{SummarizationConfig, SummarizationModel};
    use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
    use rust_bert::pipelines::zero_shot_classification::{
        ZeroShotClassificationConfig, ZeroShotClassificationModel,
    };
    use rust_bert::resources::RemoteResource;
    use rust_bert::Device;

    /// ONNX weights for `sentence-transformers/all-MiniLM-L6-v2`; auxiliary
    /// configuration/tokenizer files are shared with the torch presets pointing
    /// at the upstream `sentence-transformers` repository.
    fn all_minilm_l6_v2_onnx_config() -> SentenceEmbeddingsConfig {
        SentenceEmbeddingsConfig {
            modules_config_resource: Box::new(RemoteResource::from_pretrained(
                SentenceEmbeddingsModulesConfigResources::ALL_MINI_LM_L6_V2,
            )),
            transformer_type: ModelType::ONNX,
            transformer_config_resource: Box::new(RemoteResource::from_pretrained(
                BertConfigResources::ALL_MINI_LM_L6_V2,
            )),
            transformer_weights_resource: Box::new(RemoteResource::new(
                "https://huggingface.co/optimum/all-MiniLM-L6-v2/resolve/10244843eba3d9e479b27a4b81c94b56d8e9f4f2/model.onnx",
                "onnx-all-minilm-l6-v2",
            )),
            pooling_config_resource: Box::new(RemoteResource::from_pretrained(
                SentenceEmbeddingsPoolingConfigResources::ALL_MINI_LM_L6_V2,
            )),
            dense_config_resource: None,
            dense_weights_resource: None,
            sentence_bert_config_resource: Box::new(RemoteResource::from_pretrained(
                SentenceEmbeddingsConfigResources::ALL_MINI_LM_L6_V2,
            )),
            tokenizer_config_resource: Box::new(RemoteResource::from_pretrained(
                SentenceEmbeddingsTokenizerConfigResources::ALL_MINI_LM_L6_V2,
            )),
            tokenizer_vocab_resource: Box::new(RemoteResource::new(
                "https://huggingface.co/optimum/all-MiniLM-L6-v2/resolve/10244843eba3d9e479b27a4b81c94b56d8e9f4f2/vocab.txt",
                "onnx-all-minilm-l6-v2",
            )),
            tokenizer_merges_resource: None,
            device: Device::cuda_if_available(),
            #[cfg(feature = "libtorch")]
            kind: None,
        }
    }

    /// Builds the ONNX sentence embeddings model with an explicit tokenizer: the
    /// generic `ModelType::ONNX` has no default tokenizer, the underlying
    /// architecture (BERT, lower-cased WordPiece) must be provided.
    fn all_minilm_l6_v2_onnx_model() -> anyhow::Result<SentenceEmbeddingsModel> {
        let config = all_minilm_l6_v2_onnx_config();
        let vocab_path = config.tokenizer_vocab_resource.get_local_path()?;
        let tokenizer = TokenizerOption::from_file(
            ModelType::Bert,
            vocab_path.to_str().unwrap(),
            None,
            true,
            None,
            None,
        )?;
        Ok(SentenceEmbeddingsModel::new_with_tokenizer(
            config, tokenizer,
        )?)
    }

    #[test]
    fn onnx_sentence_embeddings() -> anyhow::Result<()> {
        let model = all_minilm_l6_v2_onnx_model()?;

        assert_eq!(model.get_embedding_dim()?, 384);

        let sentences = ["This is an example sentence", "Each sentence is converted"];
        let embeddings = model.encode(&sentences)?;

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
        assert_eq!(embeddings[1].len(), 384);

        let expected_first = [
            0.067_656_875_f32,
            0.063_495_934,
            0.048_713_12,
            0.079_304_95,
            0.037_448_026,
            0.002_652_822,
        ];
        for (actual, expected) in embeddings[0][..expected_first.len()]
            .iter()
            .zip(expected_first)
        {
            assert!((actual - expected).abs() < 1e-4);
        }

        let expected_second = [
            0.086_438_56_f32,
            0.102_762_654,
            0.005_394_486_2,
            0.002_044_387_9,
            -0.009_963_338,
            0.025_385_45,
        ];
        for (actual, expected) in embeddings[1][..expected_second.len()]
            .iter()
            .zip(expected_second)
        {
            assert!((actual - expected).abs() < 1e-4);
        }
        Ok(())
    }

    #[test]
    fn onnx_keywords_extraction_cosine_similarity() -> anyhow::Result<()> {
        let config = KeywordExtractionConfig {
            sentence_embeddings_config: all_minilm_l6_v2_onnx_config(),
            tokenizer_stopwords: None,
            tokenizer_pattern: None,
            tokenizer_forbidden_ngram_chars: None,
            scorer_type: KeywordScorerType::CosineSimilarity,
            ngram_range: (1, 2),
            num_keywords: 5,
            diversity: None,
            max_sum_candidates: None,
        };
        let sentence_embeddings_model = all_minilm_l6_v2_onnx_model()?;
        let keyword_extraction_model = KeywordExtractionModel::new_with_sentence_embeddings_model(
            config,
            sentence_embeddings_model,
        )?;

        let input = "Rust is a multi-paradigm, general-purpose programming language. \
               Rust emphasizes performance, type safety, and concurrency. Rust enforces memory safety—that is, \
               that all references point to valid memory—without requiring the use of a garbage collector or \
               reference counting present in other memory-safe languages. To simultaneously enforce \
               memory safety and prevent concurrent data races, Rust's borrow checker tracks the object lifetime \
               and variable scope of all references in a program during compilation. Rust is popular for \
               systems programming but also offers high-level features including functional programming constructs.";

        let output = keyword_extraction_model.predict(&[input])?;
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 5);
        let expected_keywords = [
            ("rust enforces", 0.647_152_7_f32),
            ("rust", 0.509_106_16),
            ("programming language", 0.486_822),
            ("rust emphasizes", 0.486_644_95),
            ("rust's", 0.478_453_55),
        ];
        for (keyword, (expected_text, expected_score)) in
            output[0].iter().zip(expected_keywords.iter())
        {
            assert_eq!(keyword.text, *expected_text);
            assert!((keyword.score - expected_score).abs() < 1e-4);
        }
        Ok(())
    }

    #[test]
    fn onnx_keywords_extraction_maximal_margin_relevance() -> anyhow::Result<()> {
        let config = KeywordExtractionConfig {
            sentence_embeddings_config: all_minilm_l6_v2_onnx_config(),
            tokenizer_stopwords: None,
            tokenizer_pattern: None,
            tokenizer_forbidden_ngram_chars: None,
            scorer_type: KeywordScorerType::MaximalMarginRelevance,
            ngram_range: (1, 2),
            num_keywords: 5,
            diversity: Some(0.5),
            max_sum_candidates: None,
        };
        let sentence_embeddings_model = all_minilm_l6_v2_onnx_model()?;
        let keyword_extraction_model = KeywordExtractionModel::new_with_sentence_embeddings_model(
            config,
            sentence_embeddings_model,
        )?;

        let input = "Rust is a multi-paradigm, general-purpose programming language. \
               Rust emphasizes performance, type safety, and concurrency. Rust enforces memory safety—that is, \
               that all references point to valid memory—without requiring the use of a garbage collector or \
               reference counting present in other memory-safe languages. To simultaneously enforce \
               memory safety and prevent concurrent data races, Rust's borrow checker tracks the object lifetime \
               and variable scope of all references in a program during compilation. Rust is popular for \
               systems programming but also offers high-level features including functional programming constructs.";

        let output = keyword_extraction_model.predict(&[input])?;
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 5);
        let expected_keywords = [
            ("rust enforces", 0.647_152_7_f32),
            ("programming language", 0.486_822),
            ("memory safety", 0.421_752_72),
            ("general-purpose", 0.316_534_22),
            ("borrow checker", 0.272_965_43),
        ];
        for (keyword, (expected_text, expected_score)) in
            output[0].iter().zip(expected_keywords.iter())
        {
            assert_eq!(keyword.text, *expected_text);
            assert!((keyword.score - expected_score).abs() < 1e-4);
        }
        Ok(())
    }

    #[test]
    fn onnx_conversation() -> anyhow::Result<()> {
        // No Optimum-layout DialoGPT export is available on the hub; the ONNX
        // conversation path is exercised with the equivalent GPT2 causal export
        // (same architecture, greedy decoding for determinism).
        let conversation_model = ConversationModel::new(ConversationConfig {
            model_type: ModelType::GPT2,
            model_resource: ModelResource::ONNX(ONNXModelResources {
                decoder_resource: Some(Box::new(RemoteResource::new(
                    "https://huggingface.co/optimum/gpt2/resolve/1869d03cd2091ba24cde0958690105c92c1ca36c/decoder_model.onnx",
                    "onnx-gpt2",
                ))),
                decoder_with_past_resource: Some(Box::new(RemoteResource::new(
                    "https://huggingface.co/optimum/gpt2/resolve/1869d03cd2091ba24cde0958690105c92c1ca36c/decoder_with_past_model.onnx",
                    "onnx-gpt2",
                ))),
                ..Default::default()
            }),
            config_resource: Box::new(RemoteResource::new(
                "https://huggingface.co/optimum/gpt2/resolve/1869d03cd2091ba24cde0958690105c92c1ca36c/config.json",
                "onnx-gpt2",
            )),
            vocab_resource: Box::new(RemoteResource::new(
                "https://huggingface.co/gpt2/resolve/main/vocab.json",
                "onnx-gpt2",
            )),
            merges_resource: Some(Box::new(RemoteResource::new(
                "https://huggingface.co/gpt2/resolve/main/merges.txt",
                "onnx-gpt2",
            ))),
            do_sample: false,
            max_length: Some(20),
            ..Default::default()
        })?;

        let mut conversation_manager = ConversationManager::new();
        let conversation_id =
            conversation_manager.create("Going to the movies tonight - any suggestions?");
        let output = conversation_model.generate_responses(&mut conversation_manager)?;

        assert_eq!(output.len(), 1);
        assert!(output.contains_key(&conversation_id));
        Ok(())
    }

    #[test]
    fn onnx_zero_shot_classification() -> anyhow::Result<()> {
        let zero_shot_model = ZeroShotClassificationModel::new(ZeroShotClassificationConfig::new(
            ModelType::DistilBert,
            ModelResource::ONNX(ONNXModelResources {
                encoder_resource: Some(Box::new(RemoteResource::new(
                    "https://huggingface.co/optimum/distilbert-base-uncased-mnli/resolve/f2a41159d38dd6a54b91839de259af674c938e69/model.onnx",
                    "onnx-distilbert-base-uncased-mnli",
                ))),
                ..Default::default()
            }),
            RemoteResource::new(
                "https://huggingface.co/optimum/distilbert-base-uncased-mnli/resolve/f2a41159d38dd6a54b91839de259af674c938e69/config.json",
                "onnx-distilbert-base-uncased-mnli",
            ),
            RemoteResource::new(
                "https://huggingface.co/optimum/distilbert-base-uncased-mnli/resolve/f2a41159d38dd6a54b91839de259af674c938e69/vocab.txt",
                "onnx-distilbert-base-uncased-mnli",
            ),
            None,
            true,
            None,
            None,
        ))?;

        let input_sentence = "Who are you voting for in 2020?";
        let input_sequence_2 =
            "The prime minister has announced a stimulus package which was widely criticized by the opposition.";
        let candidate_labels = &["politics", "public health", "economics", "sports"];

        let output = zero_shot_model.predict_multilabel(
            [input_sentence, input_sequence_2],
            candidate_labels,
            None,
            128,
        )?;
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 4);
        assert_eq!(output[1].len(), 4);

        // "Who are you voting for in 2020?" => politics
        assert_eq!(output[0][0].text, "politics");
        assert!((output[0][0].score - 0.657_651_3).abs() < 1e-4);
        assert_eq!(output[1][2].text, "economics");
        assert!((output[1][2].score - 0.499_775_05).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn onnx_summarization() -> anyhow::Result<()> {
        let summarization_model = SummarizationModel::new(SummarizationConfig::new(
            ModelType::T5,
            ModelResource::ONNX(ONNXModelResources {
                encoder_resource: Some(Box::new(RemoteResource::new(
                    "https://huggingface.co/optimum/t5-small/resolve/c5c8ea1bc3f5940643d88d792094e851fa80e06f/encoder_model.onnx",
                    "onnx-t5-small",
                ))),
                decoder_resource: Some(Box::new(RemoteResource::new(
                    "https://huggingface.co/optimum/t5-small/resolve/c5c8ea1bc3f5940643d88d792094e851fa80e06f/decoder_model.onnx",
                    "onnx-t5-small",
                ))),
                decoder_with_past_resource: Some(Box::new(RemoteResource::new(
                    "https://huggingface.co/optimum/t5-small/resolve/c5c8ea1bc3f5940643d88d792094e851fa80e06f/decoder_with_past_model.onnx",
                    "onnx-t5-small",
                ))),
            }),
            RemoteResource::new(
                "https://huggingface.co/optimum/t5-small/resolve/c5c8ea1bc3f5940643d88d792094e851fa80e06f/config.json",
                "onnx-t5-small",
            ),
            // The Optimum export does not bundle the T5 SentencePiece model; it is
            // shared from the upstream checkpoint repository.
            RemoteResource::new(
                "https://huggingface.co/google-t5/t5-small/resolve/df1b051c49625cf57a3d0d8d3863ed4d13564fe4/spiece.model",
                "t5-small",
            ),
            None,
        ))?;

        let input = ["The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. \
        Built in 1889 for the 1889 World's Fair, it was initially criticised by some of France's leading artists \
        and intellectuals for its design, but it has since become a global cultural icon of France and one of \
        the most recognisable structures in the world."];

        let output = summarization_model.summarize(&input)?;
        assert_eq!(output.len(), 1);
        assert_eq!(
            output[0],
            " the tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. built in 1889 for the 1889 world's fair, it was initially criticised by some of France's leading artists and intellectuals for its design."
        );
        Ok(())
    }

    #[test]
    fn onnx_text_generation_without_past() -> anyhow::Result<()> {
        // Decoder resource only: the generation loop falls back to running the
        // cache-less decoder at every step.
        let text_generation_model = TextGenerationModel::new(TextGenerationConfig {
            model_type: ModelType::GPT2,
            model_resource: ModelResource::ONNX(ONNXModelResources {
                decoder_resource: Some(Box::new(RemoteResource::new(
                    "https://huggingface.co/optimum/gpt2/resolve/1869d03cd2091ba24cde0958690105c92c1ca36c/decoder_model.onnx",
                    "onnx-gpt2",
                ))),
                ..Default::default()
            }),
            config_resource: Box::new(RemoteResource::new(
                "https://huggingface.co/optimum/gpt2/resolve/1869d03cd2091ba24cde0958690105c92c1ca36c/config.json",
                "onnx-gpt2",
            )),
            vocab_resource: Box::new(RemoteResource::new(
                "https://huggingface.co/gpt2/resolve/main/vocab.json",
                "onnx-gpt2",
            )),
            merges_resource: Some(Box::new(RemoteResource::new(
                "https://huggingface.co/gpt2/resolve/main/merges.txt",
                "onnx-gpt2",
            ))),
            max_length: Some(30),
            do_sample: false,
            num_beams: 1,
            temperature: 1.0,
            num_return_sequences: 1,
            ..Default::default()
        })?;

        let prompts = ["It was a very nice and sunny"];
        let output = text_generation_model.generate(&prompts, None)?;
        assert_eq!(output.len(), 1);
        assert_eq!(
            output[0],
            "It was a very nice and sunny day. I was very happy with the weather. I was very happy with the weather. I was very happy with"
        );
        Ok(())
    }
}
