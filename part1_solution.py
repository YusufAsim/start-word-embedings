import gensim.downloader
import numpy as np

# Pre-trained embedding model (do not modify)
model = gensim.downloader.load("word2vec-google-news-300")
EMBEDDING_DIM = model.vector_size


def replace_with_similar(sentence, indices):
    """
    Replace the tokens at the given indices with their top-1 most similar words.
    """
    # TODO: Implement
    pass


def sentence_vector(sentence):
    """
    Compute sentence embedding as mean of word vectors.
    """
    # TODO: Implement
    pass


def most_similar_sentences(file_path, query):
    """
    Rank sentences by cosine similarity to query.
    """
    # TODO: Implement
    pass


def analyze_dimension_contributions(word1, word2, top_k=10):
    """
    Analyze dimension-wise contributions.
    """
    # TODO: Implement
    pass
