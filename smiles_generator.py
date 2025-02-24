from rdkit import Chem
import random
import re

class RecursiveSMILESGenerator:
    def __init__(self, base_smiles=r"[*]/[N+](N[*])=C\C1=CC=CC=C1", max_depth=5):
        """
        :param base_smiles: Initial SMILES string with wildcard placeholders [*].
        :param max_depth: Maximum recursion steps allowed before finalizing the SMILES.
        """
        self.base_smiles = base_smiles
        self.max_depth = max_depth
        # Define only branching fragments that contain additional [*] placeholders.
        self.functional_groups = [
            # Carbon-based branching fragments:
            "[*]C([*])[*]",  # Tertiary carbon fragment with 3 [*] groups
            "[*]C([*])",     # Secondary carbon fragment with 2 [*] groups

            # Oxygen-based branching fragments:
            "[*]O[*]",       # Ether-like fragment (oxygen with two [*] attachments)

            # Nitrogen-based branching fragments:
            "[*]N([*])[*]",  # Tertiary amine fragment with 3 [*] attachments
            "[*]N([*])",     # Secondary amine fragment with 2 [*] attachments

            # Sulfur-based branching fragments:
            "[*]S([*])[*]",  # Sulfur fragment with 3 [*] attachments
            "[*]S([*])",     # Sulfur fragment with 2 [*] attachments

            # Phosphorus-based branching fragments:
            "[*]P([*])[*]",  # Phosphorus fragment with 3 [*] attachments
            "[*]P([*])",     # Phosphorus fragment with 2 [*] attachments
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
        Replace any remaining wildcard placeholders [*] with explicit hydrogen ([H]).
        This "caps" off the molecule so no wildcards remain.
        """
        return smiles.replace("[*]", "[H]")

    def modify_r_groups(self, current_smiles=None, depth=1):
        """
        Recursively replace the wildcard placeholders [*] in a SMILES string.
        When the recursion depth exceeds max_depth, finalize by replacing remaining [*] with [H].
        """
        if current_smiles is None:
            current_smiles = self.base_smiles

        # Stop recursion when maximum depth is exceeded.
        if depth > self.max_depth:
            return self.finalize_smiles(current_smiles)

        # Try several times to obtain a valid substitution.
        for _ in range(10):
            # Randomly choose one of the branching fragments and replace a randomly selected [*].
            mod = random.choice(self.functional_groups)
            modified_smiles = self.replace_random_occurrence(current_smiles, "[*]", mod)

            # If there are still wildcard placeholders, process them recursively.
            if "[*]" in modified_smiles:
                result = self.modify_r_groups(modified_smiles, depth + 1)
                if result is None:
                    continue  # Skip this attempt if recursive substitution failed.
                modified_smiles = result

            # Finalize by capping off any lingering [*] with [H].
            finalized = self.finalize_smiles(modified_smiles)

            # Only return the finalized SMILES if it is valid and free of any [*] placeholders.
            if self.is_valid_smiles(finalized) and "[*]" not in finalized:
                return finalized

        # Return None if no valid structure is found after several attempts.
        return None

    def generate_multiple_smiles(self, num_samples=10):
        """Generate multiple valid SMILES strings, ensuring all [*] placeholders are replaced."""
        smiles_list = []
        while len(smiles_list) < num_samples:
            smiles = self.modify_r_groups()
            if smiles:
                smiles_list.append(smiles)
        return smiles_list
