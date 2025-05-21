from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib


class EnergyPredictor:
    """Predicts energy values for molecular structures"""

    def __init__(self):
        self.model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, random_state=42)
        )

    def train(self, X, y):
        """Train multi-output regression model"""
        self.model.fit(X, y)

    def predict(self, smiles_list):
        if not hasattr(self.model, "estimators_"):
            raise RuntimeError("Model not trained! Generate data and train first.")
        X = self._extract_features(smiles_list)
        return self.model.predict(X).tolist()  # Convert to Python list for easier handling

    def _extract_features(self, smiles_list):
        """Extract features from SMILES strings"""
        features = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                features.append([0.0])  # Default for invalid molecules
                continue
            features.append([Descriptors.MolWt(mol)])  # Only molecular weight
        return features

    def save(self, path="energy_predictor.joblib"):
        """Save trained model to file"""
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        """Load trained model from file"""
        predictor = cls()
        predictor.model = joblib.load(path)
        return predictor