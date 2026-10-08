//! M2M100 source/target language pre-sets.

use crate::pipelines::translation::Language;

/// # M2M100 source languages pre-sets
pub struct M2M100SourceLanguages;

/// # M2M100 target languages pre-sets
pub type M2M100TargetLanguages = M2M100SourceLanguages;

#[rustfmt::skip]
impl M2M100SourceLanguages {
    pub const M2M100_418M: [Language; 100] = [Language::Afrikaans, Language::Danish, Language::Dutch, Language::German, Language::English, Language::Icelandic, Language::Luxembourgish, Language::Norwegian, Language::Swedish, Language::WesternFrisian, Language::Yiddish, Language::Asturian, Language::Catalan, Language::French, Language::Galician, Language::Italian, Language::Occitan, Language::Portuguese, Language::Romanian, Language::Spanish, Language::Belarusian, Language::Bosnian, Language::Bulgarian, Language::Croatian, Language::Czech, Language::Macedonian, Language::Polish, Language::Russian, Language::Serbian, Language::Slovak, Language::Slovenian, Language::Ukrainian, Language::Estonian, Language::Finnish, Language::Hungarian, Language::Latvian, Language::Lithuanian, Language::Albanian, Language::Armenian, Language::Georgian, Language::Greek, Language::Breton, Language::Irish, Language::ScottishGaelic, Language::Welsh, Language::Azerbaijani, Language::Bashkir, Language::Kazakh, Language::Turkish, Language::Uzbek, Language::Japanese, Language::Korean, Language::Vietnamese, Language::ChineseMandarin, Language::Bengali, Language::Gujarati, Language::Hindi, Language::Kannada, Language::Marathi, Language::Nepali, Language::Oriya, Language::Panjabi, Language::Sindhi, Language::Sinhala, Language::Urdu, Language::Tamil, Language::Cebuano, Language::Iloko, Language::Indonesian, Language::Javanese, Language::Malagasy, Language::Malay, Language::Malayalam, Language::Sundanese, Language::Tagalog, Language::Burmese, Language::CentralKhmer, Language::Lao, Language::Thai, Language::Mongolian, Language::Arabic, Language::Hebrew, Language::Pashto, Language::Farsi, Language::Amharic, Language::Fulah, Language::Hausa, Language::Igbo, Language::Lingala, Language::Luganda, Language::NorthernSotho, Language::Somali, Language::Swahili, Language::Swati, Language::Tswana, Language::Wolof, Language::Xhosa, Language::Yoruba, Language::Zulu, Language::HaitianCreole];
    pub const M2M100_1_2B: [Language; 100] = M2M100SourceLanguages::M2M100_418M;
}
