"""
TF-IDF Keyword Extraction Utility

Usage:
    This module provides a function to extract important keywords from a corpus of text
    using the TF-IDF (Term Frequency-Inverse Document Frequency) algorithm.

Example:
    key_tokens = tf_idf(sample_corpus)

Returns:
    A list of lowercased keywords that can be used for masking in text processing tasks.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(
    corpus: list[str], top_k: int | None = None, cutoff_thres: float | None = None, **kwargs
) -> tuple[list[str], list[float]]:
    """
    Extract key tokens from a corpus using TF-IDF.

    Args:
        corpus: A list of strings, where each string is a document.
        top_k: Maximum number of tokens to return.
        **kwargs: Additional keyword arguments to pass to TfidfVectorizer.
            Examples include:
            - max_df: Ignore terms that appear in more than this fraction of documents
            - min_df: Ignore terms that appear in fewer than this number of documents
            - stop_words: List of words to remove or 'english' for built-in stopwords
            - ngram_range: Tuple (min_n, max_n) for n-grams to consider
            - lowercase: Whether to convert all text to lowercase

    Returns:
        A list of key tokens extracted from the corpus, and their corresponding TF-IDF scores.
    """
    if not corpus:
        return []

    # Default TF-IDF parameters
    tfidf_params = {
        "max_df": 0.5,  # Ignore terms that appear in more than 50% of documents
        "min_df": 2,  # Ignore terms that appear in fewer than 2 documents
        "stop_words": "english",  # Remove English stop words
        "use_idf": True,
        "ngram_range": (1, 3),  # Only consider single words (unigrams)
        "lowercase": True,  # make case-insensitive
    }

    # Update with any user-provided parameters
    tfidf_params.update(kwargs)

    # Initialize the TF-IDF vectorizer with parameters
    vectorizer = TfidfVectorizer(**tfidf_params)

    # Fit and transform the corpus
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Get feature names (tokens)
    feature_names = vectorizer.get_feature_names_out()

    # Calculate the average TF-IDF score for each token across all documents
    avg_tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)

    # Create a list of (token, score) tuples
    token_scores = list(zip(feature_names, avg_tfidf_scores))

    # Sort tokens by their TF-IDF scores in descending order
    token_scores.sort(key=lambda x: x[1], reverse=True)

    # Extract the top tokens and scores
    if top_k is None:
        if cutoff_thres is not None:  # Get top tokens with scores above threshold
            top_n = 0
            for _, score in token_scores:
                if score < cutoff_thres:
                    break
                top_n += 1
        else:  # Get all tokens
            top_n = len(token_scores)
    else:  # Get top tokens or all if fewer tokens are available
        print(f"top_k: {top_k}, len(token_scores): {len(token_scores)}")
        top_n = min(top_k, len(token_scores))
    return [token for token, score in token_scores[:top_n]], [score for token, score in token_scores[:top_n]]


if __name__ == "__main__":
    # Sample corpus of documents about AI and machine learning
    sample_corpus = [
        "Artificial intelligence is transforming how we interact with technology.",
        "Machine learning models can recognize patterns in large datasets.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret and analyze visual information.",
        "Deep learning has revolutionized the field of artificial intelligence.",
        "Reinforcement learning allows agents to learn through trial and error.",
        "Neural networks are inspired by the structure of the human brain.",
        "Transformer models have improved machine translation and text generation.",
        "GPT and BERT are popular language models based on transformer architecture.",
        "Qwen is a powerful language model developed for various NLP tasks.",
    ]

    print("=== Sample Corpus ===")
    for i, doc in enumerate(sample_corpus):
        print(f"Document {i + 1}: {doc}")
    print()

    # Test TF-IDF
    print("=== TF-IDF Key Tokens ===")
    key_tokens = tf_idf(sample_corpus, top_k=None, cutoff_thres=None)
    print(f"Top tokens: {key_tokens}")

    # Print top 3 tokens with their relative importance
    print("\nTop 3 tokens with relative importance:")
    for i, token in enumerate(key_tokens[0][:3]):
        importance = 100 - (i * 10)  # Simple relative importance scale
        print(f"{i + 1}. {token} ({importance}% relative importance)")
    print()
