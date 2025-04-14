from agentpro.agentpro.tools.base import Tool
import joblib
import pandas as pd
import numpy as np
from typing import Optional, Any

class DiabetesPredictionTool(Tool):
    # Declare fields as optional with default None
    impute_means: Optional[dict] = None
    scaler: Optional[Any] = None
    model: Optional[Any] = None
    feature_names: Optional[list] = None
    columns_to_impute: Optional[list] = None

    def __init__(self, **data):
        super().__init__(**data)  # Initialize base class with name, description, arg
        # Set internal attributes
        self.impute_means = joblib.load('impute_means.pkl')
        self.scaler = joblib.load('scaler.pkl')
        self.model = joblib.load('model.pkl')
        self.feature_names = joblib.load('feature_names.pkl')
        self.columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    def run(self, input_data):
        if len(input_data) != 8:
            return "Error: Please provide exactly 8 values for prediction."
        input_data = list(input_data)
        for i, col in enumerate(self.feature_names):
            if col in self.columns_to_impute and input_data[i] == 0:
                input_data[i] = self.impute_means[col]
        input_df = pd.DataFrame([input_data], columns=self.feature_names)
        std_data = self.scaler.transform(input_df)
        prediction = self.model.predict(std_data)
        return 'The person is not diabetic' if prediction[0] == 0 else 'The person is diabetic'