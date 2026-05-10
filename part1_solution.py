import gensim.downloader
import numpy as np

# Pre-trained embedding model (do not modify)
model = gensim.downloader.load("word2vec-google-news-300")
EMBEDDING_DIM = model.vector_size


def replace_with_similar(sentence, indices):
    """
    Replace the tokens at the given indices with their top-1 most similar words.
    """
    tokens = sentence.split()
    replaced_tokens = tokens.copy()
    similar_dict = {}

    for index in indices:
        word = tokens[index]
        top_similar = model.most_similar(word, topn=5)
        similar_dict[word] = top_similar
        replaced_tokens[index] = top_similar[0][0]

    new_sentence = " ".join(replaced_tokens)
    return new_sentence, similar_dict


def sentence_vector(sentence):
    """
    Compute sentence embedding as mean of word vectors.
    """
    tokens = sentence.split()
    vector_dict = {}
    word_vectors = []

    for token in tokens:
        if token in model:
            vector = model[token]
        else:
            vector = np.zeros(EMBEDDING_DIM)

        vector_dict[token] = vector
        word_vectors.append(vector)

    if len(word_vectors) == 0:
        sentence_vec = np.zeros(EMBEDDING_DIM)
    else:
        sentence_vec = np.sum(word_vectors, axis=0) / len(word_vectors)

    return vector_dict, sentence_vec


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
