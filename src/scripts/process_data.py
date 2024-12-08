import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from google.cloud import storage
import pandas as pd
from io import StringIO
sns.set_theme(style="whitegrid")


def read_file(gcs_bucket_name, gcs_filename):
    """
    Read a CSV file from GCP storage bucket.
    This function connects to Google Cloud Storage, extract the specified CSV file,
    "credictcard.csv" and loads it into a Pandas DataFrame.

    Parameters
    ----------
    gcs_bucket_name : str
        The name of the Google Cloud Storage bucket.
    gcs_filename : str
        The name of the file within the bucket.

    Returns
    -------
    pd.DataFrame
        The contents of "credictcard.csv".
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket_name)
    blob = bucket.blob(gcs_filename)
    csv_content = blob.download_as_text()
    file = StringIO(csv_content)
    data = pd.read_csv(file, sep=",")
    return data


def explore_data(data):
    """
    Perform exploratory data analysis on the dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data transactions.

    Returns
    -------
    tuple
        A tuple containing:
        - pd.DataFrame: Info summary of the DataFrame.
        - pd.DataFrame: Descriptive statistics of the DataFrame.
        - pd.Series: Frequency count of each class in the 'Class' column.
    """
    data_info = data.info()
    data_describe = data.describe()
    data_class_frequency = data['Class'].value_counts()
    features = data.columns[:-1]

    # Histogram plots (univariate analysis)
    figure, axes = plt.subplots(4, 9, figsize=(15, 60))
    axes = axes.flatten()
    for j, feature in enumerate(features):
        sns.histplot(data[feature].values, bins=30, color="c", kde=True, ax=axes[j])
        axes[j].set_title(f"Histogram of {feature}")
        axes[j].set_xlabel(feature)

    plt.show()
    return data_info, data_describe, data_class_frequency


def transform_data(data):
    """
    Transform features in the DataFrame to reduce skewness.

    This function applies a log transformation to features in the DataFrame
    that have a skewness greater than 2.

    Parameters
    ----------
    data : pd.DataFrame
        The original dataframe.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with transformed features.
    """
    data_transformed = data.copy()
    # Applying log transform to reduce skewness of some features
    features = data.columns[:-1]
    for feature in features:
        if np.abs(data[feature].skew()) > 2:
            # Apply log(1+x) to avoid issues with log(0)
            data_transformed[feature] = np.sign(data[feature].values) * \
                                        np.log1p(np.abs(data[feature].values))
    return data_transformed


def process_data(gcs_bucket_name, gcs_filename):
    """
    Process the transaction data by reading from Google Cloud Storage and transforming it.

    Parameters
    ----------
    gcs_bucket_name : str
        The name of the Google Cloud Storage bucket.
    gcs_filename : str
        The name of the file within the bucket.

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame ready for modeling.
    """
    data = read_file(gcs_bucket_name, gcs_filename)
    # data_info, data_describe, data_class_frequency = explore_data(data)  # Uncomment if exploration is needed
    data_transformed = transform_data(data)
    return data_transformed
