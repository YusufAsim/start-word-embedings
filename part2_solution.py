import numpy as np
import pandas as pd
import gensim.downloader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Pre-trained embedding model (do not modify)
model = gensim.downloader.load("word2vec-google-news-300")
EMBEDDING_DIM = 300


class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x.squeeze(-1)


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
    tokens = text.split()
    vectors = []

    for token in tokens:
        if token in model:
            vectors.append(model[token])

    if len(vectors) == 0:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    average_vector = np.mean(vectors, axis=0)
    return average_vector.astype(np.float32)


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
    df = pd.read_csv(file_path)
    X = []
    y = []

    for _, row in df.iterrows():
        review = str(row["review"])
        sentiment = row["sentiment"]

        X.append(document_vector(review))

        if sentiment == "positive":
            y.append(1)
        else:
            y.append(0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


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
    torch.manual_seed(42)

    train_features = torch.tensor(X_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.float32)

    train_data = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    clf = SentimentClassifier(EMBEDDING_DIM, hidden_dim)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=learning_rate)

    clf.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = clf(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

    return clf

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
    test_features = torch.tensor(X_test, dtype=torch.float32)

    clf.eval()
    with torch.no_grad():
        outputs = clf(test_features)
        probabilities = torch.sigmoid(outputs)
        predicted_labels = (probabilities >= 0.5).to(torch.int64).cpu().numpy()

    y_true = np.asarray(y_test, dtype=np.int64)
    y_pred = np.asarray(predicted_labels, dtype=np.int64)

    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (true_positive + true_negative) / len(y_true)
    if true_positive + false_positive == 0:
        precision = 0.0
    else:
        precision = true_positive / (true_positive + false_positive)

    if true_positive + false_negative == 0:
        recall = 0.0
    else:
        recall = true_positive / (true_positive + false_negative)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
