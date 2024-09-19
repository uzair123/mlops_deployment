import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score
from starter.ml.model import train_model, compute_model_metrics, inference

class TestMLModel(unittest.TestCase):

    def setUp(self):
        # Mock training data
        self.X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y_train = np.array([0, 1, 0, 1])

        # Mock testing data for inference
        self.X_test = np.array([[1, 2], [5, 6]])

        # Mock predictions and true values for metric computation
        self.y_true = np.array([0, 1, 0, 1])
        self.y_preds = np.array([0, 1, 1, 1])

    def test_train_model(self):
        """
        Test that the train_model function returns a trained model.
        """
        model = train_model(self.X_train, self.y_train)
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertTrue(hasattr(model, 'predict'))  # Check if model has predict method

    def test_compute_model_metrics(self):
        """
        Test that compute_model_metrics returns correct precision, recall, and F1.
        """
        precision, recall, fbeta = compute_model_metrics(self.y_true, self.y_preds)

        # Calculate the expected values
        expected_precision = precision_score(self.y_true, self.y_preds, zero_division=1)
        expected_recall = recall_score(self.y_true, self.y_preds, zero_division=1)
        expected_fbeta = fbeta_score(self.y_true, self.y_preds, beta=1, zero_division=1)

        self.assertAlmostEqual(precision, expected_precision)
        self.assertAlmostEqual(recall, expected_recall)
        self.assertAlmostEqual(fbeta, expected_fbeta)

    def test_inference(self):
        """
        Test that inference returns predictions from the model.
        """
        model = train_model(self.X_train, self.y_train)  # Train the model first
        preds = inference(model, self.X_test)

        # We expect an array of predictions
        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(len(preds), len(self.X_test))


if __name__ == '__main__':
    unittest.main()
