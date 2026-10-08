/// Derived from https://github.com/MaartenGr/KeyBERT, shared under MIT License
///
/// Copyright (c) 2020, Maarten P. Grootendorst
/// Copyright (c) 2022, Guillaume Becquin
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in all
/// copies or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
/// SOFTWARE.
use crate::pipelines::keywords_extraction::KeywordScorerType;
use ndarray::{Array1, Array2, Axis};
use std::cmp::{max, min};

impl KeywordScorerType {
    pub(crate) fn score_keywords(
        &self,
        document_embedding: Array2<f32>,
        word_embeddings: Array2<f32>,
        num_keywords: usize,
        diversity: Option<f64>,
        max_sum_candidates: Option<usize>,
    ) -> Vec<(usize, f32)> {
        match self {
            KeywordScorerType::CosineSimilarity => {
                cosine_similarity_score(&document_embedding, &word_embeddings, num_keywords)
            }
            KeywordScorerType::MaximalMarginRelevance => maximal_margin_relevance_score(
                &document_embedding,
                &word_embeddings,
                num_keywords,
                diversity.unwrap_or(0.5),
            ),
            KeywordScorerType::MaxSum => {
                let num_keywords_candidates = word_embeddings.nrows();
                max_sum_score(
                    &document_embedding,
                    &word_embeddings,
                    num_keywords,
                    min(
                        max_sum_candidates.unwrap_or(num_keywords * 2),
                        num_keywords_candidates,
                    ),
                )
            }
        }
    }
}

/// L2-normalizes each row.
fn normalize_rows(array: &Array2<f32>) -> Array2<f32> {
    let mut output = array.clone();
    for mut row in output.rows_mut().into_iter() {
        let norm = row.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-12);
        for value in row.iter_mut() {
            *value /= norm;
        }
    }
    output
}

/// Cosine similarity of each row of `embeddings` against each row of `reference`
/// (or against the other rows of `embeddings` when `reference` is `None`).
/// Returns an (n_reference, n_embeddings) matrix.
fn cosine_similarity(
    document_embedding: Option<&Array2<f32>>,
    word_embeddings: &Array2<f32>,
) -> Array2<f32> {
    let normalized_words = normalize_rows(word_embeddings);
    let normalized_reference = document_embedding.map(normalize_rows);
    let reference = normalized_reference.as_ref().unwrap_or(&normalized_words);

    // (n_ref, hidden) x (hidden, n_words)
    let mut output = Array2::<f32>::zeros((reference.nrows(), normalized_words.nrows()));
    for (out_row, ref_row) in reference.rows().into_iter().enumerate() {
        for (out_col, word_row) in normalized_words.rows().into_iter().enumerate() {
            output[[out_row, out_col]] = ref_row.dot(&word_row);
        }
    }
    output
}

fn cosine_similarity_score(
    document_embedding: &Array2<f32>,
    word_embeddings: &Array2<f32>,
    num_keywords: usize,
) -> Vec<(usize, f32)> {
    let similarities = cosine_similarity(Some(document_embedding), word_embeddings)
        .into_shape((Array1::<f32>::zeros(word_embeddings.nrows()).len(),))
        .map(|array| array.to_owned())
        .unwrap_or_else(|_| ndarray::Array1::zeros(word_embeddings.nrows()));

    let mut order: Vec<usize> = (0..similarities.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        similarities[b]
            .partial_cmp(&similarities[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order
        .into_iter()
        .take(num_keywords)
        .map(|pos| (pos, similarities[pos]))
        .collect()
}

fn maximal_margin_relevance_score(
    document_embedding: &Array2<f32>,
    word_embeddings: &Array2<f32>,
    num_keywords: usize,
    diversity: f64,
) -> Vec<(usize, f32)> {
    let word_document_similarities_full =
        cosine_similarity(Some(document_embedding), word_embeddings);
    let word_document_similarities =
        ndarray::Array1::from(word_document_similarities_full.column(0).to_vec());
    let word_similarities = cosine_similarity(None, word_embeddings);

    let mut keyword_indices = vec![argmax_1d(&word_document_similarities)];
    let mut candidate_indices: Vec<usize> = (0..word_embeddings.nrows()).collect();
    candidate_indices.remove(keyword_indices[0] as usize);
    for _ in 0..min(num_keywords - 1, word_embeddings.nrows()) {
        let mut best_index = 0usize;
        let mut best_mmr = f32::NEG_INFINITY;
        for &candidate in &candidate_indices {
            let candidate_similarity = word_document_similarities[candidate];
            let mut target_similarity = f32::NEG_INFINITY;
            for &keyword in &keyword_indices {
                target_similarity = target_similarity.max(word_similarities[[candidate, keyword]]);
            }
            let mmr = candidate_similarity as f32 * (1.0 - diversity as f32)
                - target_similarity * diversity as f32;
            if mmr > best_mmr {
                best_mmr = mmr;
                best_index = candidate;
            }
        }
        keyword_indices.push(best_index);
        let candidate_mmr_index = candidate_indices
            .iter()
            .position(|x| *x == best_index)
            .unwrap();
        candidate_indices.remove(candidate_mmr_index);
    }

    keyword_indices
        .into_iter()
        .map(|index| (index as usize, word_document_similarities[index] as f32))
        .collect()
}

fn argmax_1d(array: &Array1<f32>) -> usize {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in array.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best = index;
        }
    }
    best
}

fn max_sum_score(
    document_embedding: &Array2<f32>,
    word_embeddings: &Array2<f32>,
    num_keywords: usize,
    max_sum_candidates: usize,
) -> Vec<(usize, f32)> {
    let max_sum_candidates = max(num_keywords, max_sum_candidates);
    let word_document_similarities_full =
        cosine_similarity(Some(document_embedding), word_embeddings);
    let word_document_similarities =
        ndarray::Array1::from(word_document_similarities_full.column(0).to_vec());
    let word_similarities = cosine_similarity(None, word_embeddings);

    let mut order: Vec<usize> = (0..word_document_similarities.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        word_document_similarities[b]
            .partial_cmp(&word_document_similarities[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_keywords: Vec<usize> = order.into_iter().take(max_sum_candidates).collect();

    let mut best_score: Option<f64> = None;
    let mut best_combination: Option<Vec<usize>> = None;
    // enumerate combinations of num_keywords among top candidates
    let n = top_keywords.len();
    let k = num_keywords;
    let mut combination = vec![0usize; k];
    let mut generate = |combination: &mut Vec<usize>| -> bool {
        // next lexicographic combination
        let mut i = k;
        while i > 0 && combination[i - 1] == n - k + i - 1 {
            i -= 1;
        }
        if i == 0 {
            return false;
        }
        combination[i - 1] += 1;
        for j in i..k {
            combination[j] = combination[j - 1] + 1;
        }
        true
    };
    // initialize 0..k
    for (i, value) in combination.iter_mut().enumerate() {
        *value = i;
    }
    loop {
        let combination_score: f64 = (0..k)
            .map(|a| {
                (0..k)
                    .map(|b| {
                        word_similarities
                            [[top_keywords[combination[a]], top_keywords[combination[b]]]]
                            as f64
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
            / 2f64;
        if let Some(current_best_score) = best_score {
            if combination_score < current_best_score {
                best_score = Some(combination_score);
                best_combination = Some(
                    top_keywords
                        .iter()
                        .map(|&i| i)
                        .collect::<Vec<usize>>()
                        .iter()
                        .map(|&i| i)
                        .collect(),
                );
                best_combination = Some(combination.iter().map(|&i| top_keywords[i]).collect());
            }
        } else {
            best_score = Some(combination_score);
            best_combination = Some(combination.iter().map(|&i| top_keywords[i]).collect());
        }
        if !generate(&mut combination) {
            break;
        }
    }

    best_combination
        .unwrap()
        .into_iter()
        .map(|index| (index, word_document_similarities[index] as f32))
        .collect()
}
