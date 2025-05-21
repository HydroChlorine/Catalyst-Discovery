from smiles_generator import RecursiveSMILESGenerator
from database import CatalystDatabase
import numpy as np


class RLGenerator(RecursiveSMILESGenerator):
    def __init__(self, db, model, max_diff=0.04, **kwargs):  # Add max_diff parameter
        super().__init__(**kwargs)
        self.db = db
        self.model = model
        self.max_diff = max_diff  # Use parameter instead of hardcoded value

    def generate_optimized(self, n=10):
        candidates = []
        for _ in range(n * 2):
            smi = self.generate_smiles()
            if smi and not self.db.exists(smi):
                candidates.append(smi)

        if not candidates:
            return []

        # Get predictions and calculate differences
        try:
            predictions = self.model.predict(candidates)
            diffs = self._calculate_differences(predictions)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

        # Filter and score based on new criteria
        valid = []
        for i, (smi, preds) in enumerate(zip(candidates, predictions)):
            current_diff = diffs[i]
            if self._is_valid(current_diff):
                valid.append((smi, preds, current_diff))

        # Sort by combined score
        sorted_candidates = sorted(valid, key=lambda x: self._score(x[2]))
        return sorted_candidates[:n]

    def _calculate_differences(self, predictions):
        return [(p[1] - p[0], p[4] - p[3]) for p in predictions]

    def _is_valid(self, diff_pair):
        return all(abs(d) <= self.max_diff for d in diff_pair)

    def _score(self, diff_pair):
        return sum(abs(d) for d in diff_pair)