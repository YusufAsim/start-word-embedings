"""
Sample input script for Part II.

This file is provided to show the expected function calls and output format.
It is not the hidden test script.

Before running this file:
1. Put this file in the same folder as your solution file.
2. Put subset10000_IMDB_Dataset.csv in the same folder.
3. Rename the import below according to your solution file name.

The dataset file should contain:
review,sentiment
"""

from sklearn.model_selection import train_test_split

from part2_solution import (
    document_vector,
    prepare_classification_data,
    train_classifier,
    evaluate_classifier,
)

DATA_PATH = "subset10000_IMDB_Dataset.csv"

with open("sample_output_part2_generated.txt", "w", encoding="utf-8") as f:
    sample_text = "This movie was great and very enjoyable"
    vec = document_vector(sample_text)

    print("Document vector length:", len(vec), file=f)
    print("First 5 values:", vec[:5], end="\n\n", file=f)

    X, y = prepare_classification_data(DATA_PATH)

    print("X shape:", X.shape, file=f)
    print("y shape:", y.shape, file=f)
    print("Number of positive labels:", int((y == 1).sum()), file=f)
    print("Number of negative labels:", int((y == 0).sum()), end="\n\n", file=f)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    clf = train_classifier(
        X_train,
        y_train,
        hidden_dim=64,
        learning_rate=0.001,
        epochs=5,
        batch_size=32
    )

    results = evaluate_classifier(clf, X_test, y_test)

    print("Evaluation results on the sample subset:", file=f)
    print(results, file=f)
