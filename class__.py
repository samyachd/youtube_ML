from abc import ABC
import joblib

class AbstractPredictor(ABC):

    def predict():
        pass

class Predictor(AbstractPredictor):

    def __init__(self, model_path, scaler_path=None):

        self.model_path = model_path
        
        self.scaler_path = scaler_path

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    