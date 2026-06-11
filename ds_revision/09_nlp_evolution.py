"""
Lab 09: Natural Language Processing - Deep Dive Tutorial
=======================================================
Syllabus Topics: BoW, TF-IDF, Word Embeddings (Word2Vec), Contextual Embeddings (Transformers).

This script explains the "evolution" of how computers read text.
From counting words to understanding deep semantic meaning.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# 1. THE BASICS: BAG OF WORDS (BoW)
# =============================================================================
# Problem: How do we turn a sentence into a vector?
# Solution: Create a "Vocabulary" of all words. For each doc, count the occurrences.
# Drawback: "This movie is good" and "This movie is NOT good" look almost the same.
# It loses context and word order.

corpus = [
    "Data science is the future of data.",
    "Machine learning learns from data.",
    "The future is full of artificial intelligence.",
]

bow_vec = CountVectorizer(stop_words="english")
bow_matrix = bow_vec.fit_transform(corpus)
print("--- Bag of Words (BoW) ---")
print(f"Vocab: {bow_vec.get_feature_names_out()}")

# =============================================================================
# 2. THE IMPROVEMENT: TF-IDF
# =============================================================================
# Problem with BoW: Common words like "the", "is", "data" get high scores but
# don't carry much meaning.
# - TF (Term Frequency): How often a word appears in a doc.
# - IDF (Inverse Document Frequency): How "unique" a word is across the WHOLE corpus.
# - LOGIC: If a word is rare in the corpus but common in one doc, it must be IMPORTANT.
#   TF-IDF(t, d) = TF(t, d) * log(Total Docs / Docs with word t)

tfidf_vec = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vec.fit_transform(corpus)
print("\n--- TF-IDF Weights (First Doc) ---")
# Rare words like 'future' will get higher weights than common ones like 'data'
print(
    pd.DataFrame(
        tfidf_matrix.toarray(), columns=tfidf_vec.get_feature_names_out()
    ).iloc[0]
)

# =============================================================================
# 3. DENSE EMBEDDINGS: WORD2VEC (Static Meaning)
# =============================================================================
# Problem: BoW and TF-IDF are "Sparse" (mostly zeros).
# Solution: Map every word to a DENSE vector (e.g., 300 numbers).
# - CBOW: Predict a word from its neighbors.
# - SKIP-GRAM: Predict neighbors from a word.
# CONCEPT: Semantic relationships.
# "King" and "Queen" will have very similar vectors because they appear in similar contexts.
print("\n--- Word2Vec Logic ---")
print("Words become 'locations' in a high-dimensional space.")
print("Distance between 'Pizza' and 'Burger' is small. Distance to 'Cloud' is large.")

# =============================================================================
# 4. THE REVOLUTION: TRANSFORMERS & ATTENTION (Contextual Meaning)
# =============================================================================
# Problem with Word2Vec: "Bank" has ONE vector.
# But "River bank" and "Bank account" mean different things!
# - ATTENTION: The model "looks" at surrounding words to decide the current word's meaning.
# - TRANSFORMERS (BERT/GPT): Generate dynamic embeddings.
#   The word "Bank" gets a DIFFERENT vector in every sentence.

# CONCEPT: Encoder vs Decoder
# - ENCODER (BERT): Reads the whole sentence to understand it. (Good for Sentiment, Classification)
# - DECODER (GPT): Predicts the NEXT word. (Good for Chatbots, Text generation)

print("\n--- Transformers (The Modern Era) ---")
print("Self-Attention allows every word to interact with every other word.")
print("This is why ChatGPT feels so 'smart'.")

# =============================================================================
# 5. SEMANTIC SIMILARITY
# =============================================================================
# We use COSINE SIMILARITY to see how similar two documents are based on their vectors.
sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
print(f"\nSimilarity between Doc 0 and Doc 1: {sim[0][0]:.4f}")

print("\n[SUCCESS] Lab 09 Complete. Text has been vectorized!")
