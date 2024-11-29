import os
from numpy import loadtxt
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_data(file_path):
    """
    Load dataset from a CSV file.

    :param file_path: Path to the CSV file.
    :return: Numpy array of the dataset.
    """
    return loadtxt(file_path, delimiter=",", skiprows=1)


def split_data(dataset, test_size=0.33, seed=7):
    """
    Split the dataset into features (X) and target (y), then into training and testing sets.

    :param dataset: Numpy array of the dataset.
    :param test_size: Proportion of the dataset to include in the test split.
    :param seed: Random seed for reproducibility.
    :return: X_train, X_test, y_train, y_test
    """
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    return train_test_split(X, Y, test_size=test_size, random_state=seed)


def train_model(X_train, y_train):
    """
    Train an XGBoost classifier on the training data.

    :param X_train: Features for training.
    :param y_train: Target values for training.
    :return: Trained model.
    """
    model = XGBClassifier()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test data.

    :param model: Trained model.
    :param X_test: Features for testing.
    :param y_test: Target values for testing.
    :return: Accuracy of the model.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


# Main workflow
def main():
    """
    Main function to execute the workflow.
    """
    # Load dataset
    file_path = os.path.join("data", "diabetes.csv")
    dataset = load_data(file_path)

    # Split data
    X_train, X_test, y_train, y_test = split_data(dataset)

    # Train model
    model = train_model(X_train, y_train)
    print(model)

    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


if __name__ == "__main__":
    main()
