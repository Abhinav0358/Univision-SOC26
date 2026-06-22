"""
Natural Language Processing: Text Representation Evolution

Traces text representation methods from basic Bag-of-Words (BoW) 
to TF-IDF weighting, static word embeddings (Word2Vec), and contextual 
embeddings (Transformers/Self-Attention).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Bag of Words (BoW)
# Maps documents to frequency vectors based on a global vocabulary.
# Limitations: Discards word order, syntax, and contextual changes in meaning.

corpus = [
    "Data science is the future of data.",
    "Machine learning learns from data.",
    "The future is full of artificial intelligence.",
]

bow_vec = CountVectorizer(stop_words="english")
bow_matrix = bow_vec.fit_transform(corpus)
print("--- Bag of Words (BoW) ---")
print(f"Vocabulary: {bow_vec.get_feature_names_out()}")

# 2. Term Frequency-Inverse Document Frequency (TF-IDF)
# Addresses the frequency bias of BoW by weighting terms based on uniqueness.
# - Term Frequency (TF): Occurrence count of a term within a specific document.
# - Inverse Document Frequency (IDF): Logarithmic scaling of term uniqueness across the corpus.
# Formula: TF-IDF(t, d, D) = TF(t, d) * log(N / df(t))

tfidf_vec = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vec.fit_transform(corpus)
print("\n--- TF-IDF Weights (First Document) ---")
print(
    pd.DataFrame(
        tfidf_matrix.toarray(), columns=tfidf_vec.get_feature_names_out()
    ).iloc[0]
)

# 3. Dense Word Embeddings (Word2Vec)
# Maps words to lower-dimensional dense vectors where spatial proximity encodes semantic similarity.
# - CBOW (Continuous Bag of Words): Learns weights by predicting a target word from context words.
# - Skip-Gram: Learns weights by predicting surrounding context words from a target word.
# Limitations: Static embeddings assign the same vector regardless of context.
print("\n--- Word2Vec Concepts ---")
print("Words are represented as coordinates in a continuous vector space.")
print("Semantic relationships are captured via vector addition and subtraction.")

# 4. Contextual Embeddings (Transformers & Self-Attention)
# Modern approaches utilize self-attention mechanisms to generate dynamic,
# sentence-specific word representations.
# - Self-Attention: Allows the model to weight the relevance of other words in a sequence.
# - Encoder Architectures (e.g., BERT): Read bidirectionally, useful for classification tasks.
# - Decoder Architectures (e.g., GPT): Read autoregressively, optimized for generative text tasks.
print("\n--- Transformers & Modern NLP ---")
print("Self-attention dynamically re-weights token vectors based on their immediate context.")

# 5. Semantic Similarity
# Calculates the cosine of the angle between document vectors to assess thematic similarity.
sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
print(f"\nCosine Similarity (Doc 0 vs. Doc 1): {sim[0][0]:.4f}")

print("\n[SUCCESS] NLP evolution lab completed successfully.")
