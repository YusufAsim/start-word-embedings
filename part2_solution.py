import numpy as np
import pandas as pd
import gensim.downloader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Pre-trained embedding model (do not modify)
model = gensim.downloader.load("word2vec-google-news-300")
EMBEDDING_DIM = 300


def document_vector(text):
    """
    Convert text into a single vector by averaging word embeddings.

    Requirements:
    - Tokenize the text based on whitespace.
    - Ignore out-of-vocabulary (OOV) words.
    - If all words are OOV, return a zero vector with shape (300,).

    Args:
        text (str): input review text

    Returns:
        np.ndarray: document vector with shape (300,)
    """
    # TODO: Implement
    pass


def prepare_classification_data(file_path):
    """
    Load the IMDB dataset and convert it into feature and label arrays.

    Expected CSV format:
        review,sentiment

    Label mapping:
        positive -> 1
        negative -> 0

    Args:
        file_path (str): path to the CSV file

    Returns:
        X (np.ndarray): feature matrix with shape (N, 300)
        y (np.ndarray): label vector with shape (N,)
    """
    # TODO: Implement
    pass


def train_classifier(X_train, y_train,
                     hidden_dim=64,
                     learning_rate=0.001,
                     epochs=10,
                     batch_size=32):
    """
    Train a simple neural network classifier using document vectors.

    Requirements:
    - Use a deep learning framework such as PyTorch.
    - The model must have:
        * an input layer matching the embedding dimension,
        * at least one hidden layer,
        * an output layer for binary classification.
    - The function must return the trained model.

    Args:
        X_train (np.ndarray): training feature matrix
        y_train (np.ndarray): training labels

    Returns:
        torch.nn.Module: trained neural network classifier
    """
    # TODO: Implement
    pass

def evaluate_classifier(clf, X_test, y_test):
    """
    Evaluate the trained classifier.

    Requirements:
    - Predict labels for X_test.
    - Compute accuracy, precision, recall, and F1-score.

    Args:
        clf (torch.nn.Module): trained classifier
        X_test (np.ndarray): test feature matrix
        y_test (np.ndarray): test labels

    Returns:
        dict: {
            "accuracy": float,
            "precision": float,
            "recall": float,
            "f1": float
        }
    """
    # TODO: Implement
    pass
