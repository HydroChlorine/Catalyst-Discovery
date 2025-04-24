from smiles_generator import RecursiveSMILESGenerator
from .database import CatalystDatabase
import numpy as np


class RLGenerator(RecursiveSMILESGenerator):
    """Reinforcement learning optimized structure generator"""

    def __init__(self, db, model, **kwargs):
        super().__init__(**kwargs)
        self.db = db  # Database instance
        self.model = model  # Trained prediction model
        self.target_ranges = [  # Example target energy ranges
            (-10.5, -9.5),  # e1 range
            (-5.2, -4.8),  # e2 range
            # ... other 4 energy ranges
        ]

    def generate_optimized(self, n=10):
        """Generate optimized candidate structures"""
        candidates = []
        for _ in range(n * 2):  # Generate double for filtering
            smi = self.generate_smiles()
            if smi and not self.db.exists(smi):
                candidates.append(smi)

        # Predict energies and filter
        predictions = self.model.predict(candidates)
        scored = [(smi, preds) for smi, preds in zip(candidates, predictions)
                  if self._is_predicted_valid(preds)]

        return [smi for smi, _ in sorted(scored, key=lambda x: self._score(x[1]))[:n]]

    def _is_predicted_valid(self, energies):
        """Check if predicted energies fall within target ranges"""
        return all(low <= e <= high for e, (low, high)
                   in zip(energies, self.target_ranges))

    def _score(self, energies):
        """Calculate energy match score (lower is better)"""
        return sum(
            abs(e - (high + low) / 2) for e, (low, high)  # Distance from target midpoint
            in zip(energies, self.target_ranges)
        )