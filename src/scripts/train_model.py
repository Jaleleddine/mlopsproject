import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from process_data import process_data
from utils import get_config, parse_args


def fit_model(data_transformed):
    """
    Fit an Isolation Forest model to the transformed data.
    This function preprocesses the feature data by scaling it using
    StandardScaler, then fits an Isolation Forest model.

    Parameters
    ----------
    data_transformed : pd.DataFrame
        The transformed input data containing features and the target variable.

    Returns
    -------
    IsolationForest
        The fitted Isolation Forest model.
    """
    X = data_transformed.drop(columns=['Class'])
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=10)
    X_scaled = pca.fit_transform(X_scaled)

    model = IsolationForest(n_estimators=100,
                            contamination=0.01,
                            random_state=42)
    model.fit(X_scaled)
    return model,pca


def evaluate_model(model,pca, data_transformed):
    """
    Evaluate the fitted model's performance.
    This function calculates and prints the recall, precision, and F1 score
    for the predictions made by the model on the transformed data.

    Parameters
    ----------
    model : IsolationForest
        The fitted Isolation Forest model to evaluate.

    data_transformed : pd.DataFrame
        The transformed input data containing features and the true target variable.

    Returns
    -------
    tuple
        A tuple containing:
        - recall : float
            The recall score of the model.
        - precision : float
            The precision score of the model.
        - f1s : float
            The F1 score of the model.
    """
    X = data_transformed.drop(columns=['Class'])
    X = pca.fit_transform(X)

    y_true = data_transformed["Class"].values.tolist()
    y_pred = model.predict(X)
    y_pred = [1 if i == -1 else 0 for i in y_pred]

    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1s = f1_score(y_true, y_pred)

    print("The recall rate is {rec}".format(rec=recall))
    print("The precision rate is {precision}".format(precision=precision))
    print("The F1 score rate is {f1}".format(f1=f1s))

    return recall, precision, f1s


def save_model(model, model_path):
    """
    Save the trained model to a specified file path.
    This function uses joblib to save the fitted model.

    Parameters
    ----------
    model : IsolationForest
        The fitted Isolation Forest model.

    model_path : str
        The model file path it will be saved.
    """
    joblib.dump(model, model_path)


def main(args):
    """
    Main function

    Parameters
    ----------
    args : Namespace
        Parsed command line arguments containing configuration and retrain flag.
    """
    config = get_config(args.configuration)
    retrain = args.retrain

    if retrain == "false":
        data_transformed = process_data(config["FILES"]["GCS_BUCKET_NAME"],
                                        config["FILES"]["GCS_FILE_NAME"])
    if retrain == "true":
        data_transformed = process_data(config["FILES"]["GCS_BUCKET_NAME"],
                                        config["FILES"]["SAMPLE_GCS_FILE_NAME"])

    model,pca = fit_model(data_transformed)
    recall, precision, f1s = evaluate_model(model,pca, data_transformed)
    print(recall, precision)
    save_model(model, config["FILES"]["MODEL_PATH"])


if __name__ == "__main__":
    args = parse_args()
    main(args)
