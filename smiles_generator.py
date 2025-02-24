from rdkit import Chem
import random
import re

class RecursiveSMILESGenerator:
    def __init__(self, base_smiles=r"[R]/[N+](N[R])=C\C1=CC=CC=C1", max_depth=5):
        """
        :param base_smiles: Initial SMILES string with [R] placeholders.
        :param max_depth: Maximum recursion steps allowed before finalizing the SMILES.
        """
        self.base_smiles = base_smiles
        self.max_depth = max_depth
        # Define only branching fragments that contain additional [R] placeholders.
        self.functional_groups = [
            # Carbon-based branching fragments:
            "[R]C([R])[R]",  # Tertiary carbon fragment with 3 [R] groups
            "[R]C([R])",     # Secondary carbon fragment with 2 [R] groups

            # Oxygen-based branching fragments:
            "[R]O[R]",       # Ether-like fragment (oxygen with two [R] attachments)

            # Nitrogen-based branching fragments:
            "[R]N([R])[R]",  # Tertiary amine fragment with 3 [R] attachments
            "[R]N([R])",     # Secondary amine fragment with 2 [R] attachments

            # Sulfur-based branching fragments:
            "[R]S([R])[R]",  # Sulfur fragment with 3 [R] attachments
            "[R]S([R])",     # Sulfur fragment with 2 [R] attachments

            # Phosphorus-based branching fragments:
            "[R]P([R])[R]",  # Phosphorus fragment with 3 [R] attachments
            "[R]P([R])",     # Phosphorus fragment with 2 [R] attachments
        ]

    @staticmethod
    def replace_random_occurrence(s, placeholder, replacement):
        """
        Find all occurrences of placeholder in s,
        randomly select one occurrence, and replace it with replacement.
        """
        matches = list(re.finditer(re.escape(placeholder), s))
        if not matches:
            return s
        match = random.choice(matches)
        start, end = match.span()
        return s[:start] + replacement + s[end:]

    def is_valid_smiles(self, smiles):
        """Checks if the generated SMILES is valid using RDKit."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except Exception:
            return False

    def finalize_smiles(self, smiles):
        """
        Replace any remaining [R] placeholders with explicit hydrogen ([H]).
        """
        return smiles.replace("[R]", "[H]")

    def modify_r_groups(self, current_smiles=None, depth=1):
        """
        Recursively replace the [R] groups in a SMILES string.
        When the recursion depth exceeds max_depth, finalize by replacing remaining [R] with [H].
        """
        if current_smiles is None:
            current_smiles = self.base_smiles

        # Stop recursion when the maximum depth is exceeded.
        if depth > self.max_depth:
            return self.finalize_smiles(current_smiles)

        # Try several times to obtain a valid substitution.
        for _ in range(10):
            # Use our custom replace method to randomly choose a [R] to replace.
            mod = random.choice(self.functional_groups)
            modified_smiles = self.replace_random_occurrence(current_smiles, "[R]", mod)

            # If there are still [R] placeholders, process them recursively.
            if "[R]" in modified_smiles:
                result = self.modify_r_groups(modified_smiles, depth + 1)
                if result is None:
                    continue  # Skip this attempt if recursive substitution failed.
                modified_smiles = result

            # Finalize by replacing any remaining [R] with [H].
            finalized = self.finalize_smiles(modified_smiles)

            # Only return the finalized SMILES if it is valid and free of [R] placeholders.
            if self.is_valid_smiles(finalized) and "[R]" not in finalized:
                return finalized

        return None  # Return None if no valid structure is found after several attempts.

    def generate_multiple_smiles(self, num_samples=10):
        """Generate multiple valid SMILES strings, ensuring all [R] are replaced."""
        smiles_list = []
        while len(smiles_list) < num_samples:
            smiles = self.modify_r_groups()
            if smiles:
                smiles_list.append(smiles)
        return smiles_list
