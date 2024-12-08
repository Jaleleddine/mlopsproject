import joblib
from sklearn.metrics import recall_score

from process_data import transform_data, read_file
from utils import get_config, parse_args


def load_updated_data():
    """
    Load the updated data for monitoring and modify the target variable.
    This function reads the sample data from GCP storage bucket which is used to test monitoring step,
    modifies the 'Class' target variable to 1 to simulate a scenario where the model
    is expected to be retrained.

    Returns
    -------
    pd.DataFrame
        The modified sample data with 'Class' set to 1.
    """
    sample_data = read_file(config["FILES"]["GCS_BUCKET_NAME"],
                            config["FILES"]["SAMPLE_GCS_FILE_NAME"])
    sample_data["Class"] = 1  # Change target in order to drop down the recall and test retraining new model
    return sample_data


def monitor(config):
    """
    Monitor the model's performance by evaluating its recall score.

    This function loads the existing model and new data, transforms the new data,
    and evaluates the model's predictions against the new data. If the recall score
    is below the specified threshold, it triggers a decision to retrain the model.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing file paths and parameters.

    Returns
    -------
    None
        Writes decision output to the monitoring file based on the recall score.
    """
    # Load the existing model
    model = joblib.load(config["FILES"]["MODEL_PATH"])

    # Load and transform new data
    new_data = load_updated_data()
    new_data_transformed = transform_data(new_data)

    # Predict new data with existing model and check recall
    X_new = new_data_transformed.drop(columns=['Class'])
    y_true = new_data_transformed["Class"].values.tolist()
    y_pred = model.predict(X_new)

    # Convert prediction labels
    y_pred = [1 if i == -1 else 0 for i in y_pred]

    # Calculate recall
    recall = recall_score(y_true, y_pred)

    # Send decision to retrain or not, truncate it each time
    with open(config["FILES"]["MONITORING_PATH"], 'w') as file:
        if recall < float(config["MONITOR"]["RECALL_THRESHOLD"]):
            file.write("retrain new model")
        else:
            file.write("keep existing model")


if __name__ == "__main__":
    args = parse_args()
    config = get_config(args.configuration)
    monitor(config)
