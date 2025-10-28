import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Logging Configuration ---
# Set up a logger that will output messages to the console and a file for errors
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

# Prevent adding handlers multiple times if run in interactive mode
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel('DEBUG')

    file_handler = logging.FileHandler('model_evaluation_errors.log')
    file_handler.setLevel('ERROR')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path) # RESTORED: pd.read_csv
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file: # RESTORED: with open
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file: # RESTORED: with open
            vectorizer = pickle.load(file)
        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and log classification metrics and confusion matrix."""
    try:
        # Predict and calculate classification metrics
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        logger.debug('Model evaluation completed')

        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def log_confusion_matrix(cm, dataset_name):
    """Log confusion matrix as an artifact."""
    plt.figure(figsize=(8, 6))
    # Assuming class labels are 0 and 1, adjust xticklabels/yticklabels if classes are known
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save confusion matrix plot as a file and log it to MLflow
    cm_file_path = f'confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()


def main():
    # NOTE: Set the correct public IP for your MLflow tracking server
    mlflow.set_tracking_uri("http://13.203.227.156:5000/")

    mlflow.set_experiment('dvc-pipeline')

    with mlflow.start_run():
        try:
            # Determine the project root for DVC/MLOps structure
            # Assumes this script is run from a subfolder (e.g., 'src/evaluation')
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Adjusted root path logic for better compatibility
            # This moves up two directories from the script location.
            root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

            # Load parameters from YAML file
            params = load_params(os.path.join(root_dir, 'params.yaml'))

            # Log parameters (from params.yaml)
            for key, value in params.items():
                mlflow.log_param(key, value)

            # Load model and vectorizer
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Log model parameters (from the model object itself)
            if hasattr(model, 'get_params'):
                for param_name, param_value in model.get_params().items():
                    mlflow.log_param(param_name, param_value)

            # Load test data
            test_data_path = os.path.join(root_dir, 'data/interim/test_processed.csv')
            test_data = load_data(test_data_path)

            # Prepare test data
            X_test_text = test_data['clean_comment'].values
            y_test = test_data['category'].values

            # Transform test data
            X_test_tfidf = vectorizer.transform(X_test_text)

            # Log model (using the deprecated artifact_path/name parameter, kept for compatibility)
            # NOTE: For production, consider manually logging the vectorizer as a custom artifact
            mlflow.sklearn.log_model(model, "lgbm_model")

            # Log the vectorizer as a separate artifact for clarity/deployment
            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Evaluate model and get metrics
            report, cm = evaluate_model(model, X_test_tfidf, y_test)

            # Log classification report metrics for the test data
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    # Log precision, recall, and f1-score
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })
                elif label in ['accuracy']: # Handle overall metrics like accuracy if reported by older scikit-learn
                    mlflow.log_metric(f"test_{label}", metrics)


            # Log confusion matrix
            log_confusion_matrix(cm, "Test Data")

            # Add important tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

            logger.info(f"MLflow Run completed successfully. Run ID: {mlflow.active_run().info.run_id}")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")


if __name__ == '__main__':
    main()
