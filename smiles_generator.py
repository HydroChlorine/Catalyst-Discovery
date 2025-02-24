from rdkit import Chem
import random


class RecursiveSMILESGenerator:
    def __init__(self, base_smiles=r"[R]/[N+](N[R])=C\C1=CC=CC=C1"):
        self.base_smiles = base_smiles
        # Define functional groups that can replace R, including common elements C, O, N, S, and P
        self.functional_groups = [
            # Carbon-based groups
            "C",  # Methyl group (CH3)
            "CC",  # Ethyl group (C2H5)
            "C1CCCCC1",  # Cyclohexane (C6H12)
            "C1=CC=CC=C1",  # Phenyl group (C6H5)
            "C#C",  # Alkyne group (C≡C)
            "C=C",  # Alkene group (C=C)

            # Oxygen-based groups
            "O",  # Hydroxyl group (OH)
            "CO",  # Methoxy group (OCH3)
            "C=O",  # Carbonyl group (C=O)
            "C(O)=O",  # Carboxyl group (-COOH)
            "C1CCOCC1",  # Ethoxyethane (C2H5OCH2)
            "O=C",  # Aldehyde group (C=O)
            "C(O)C",  # Alcohol with an additional alkyl group

            # Nitrogen-based groups
            "NC",  # Amine group (NH2 with CH3)
            "N",  # Amino group (NH2)
            "C1=CC=CC=C1[NH]",  # Phenylamine group (C6H5NH)
            "C1CCN(C)CC1",  # N-Methylpiperidine (C6H12)
            "[NH2]",  # Amine group (NH2, simple)
            "[N+](C)(C)[O-]",  # Nitroso group (-NO)

            # Sulfur-based groups
            "S",  # Thiol group (SH)
            "SC",  # Thiomethyl group (-SH)
            "S=O",  # Sulfoxide group (C=SO)
            "C=S",  # Thiocarbonyl group (C=S)
            "C1CCSCC1",  # Ethane-1,2-dithiol (C4H10S2)

            # Phosphorus-based groups
            "P",  # Phosphine group (PH2)
            "P=O",  # Phosphate group (-PO4)
            "P(CO)O",  # Phosphoryl group (-PO2)
            "C1CCPCC1",  # Phosphorus in a cyclic structure (C4H9P)
            "C1=CCPCC1",  # Phosphoryl group in a ring structure
            "P(O)C",  # Phosphorochloridate group (P(O)C)

            # Recursive examples: Allowing [R] to be part of the functional group itself
            "[R]O",  # Alcohol with another [R]
            "[R]N",  # Amine with another [R]
            "[R]S",  # Thiol with another [R]
            "[R]P",  # Phosphine with another [R]
            "[R]C([R])[R]",  # A carbon with 3 mutable R groups
            "[R]C([R])",  # A carbon with 2 mutable R groups and one H
            "[R]C",  # A carbon with 1 mutalbe R group and two H
        ]

    def is_valid_smiles(self, smiles):
        """Checks if the generated SMILES is valid using RDKit."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def modify_r_groups(self, current_smiles=None, depth=1):
        """Replace the R group in a SMILES string and handle recursion."""
        if current_smiles is None:
            current_smiles = self.base_smiles

        if depth > 3:  # Limiting recursion depth to avoid infinite loops
            return current_smiles

        for _ in range(10):  # Try up to 10 times to get a valid substitution
            mod = random.choice(self.functional_groups)
            modified_smiles = current_smiles.replace("[R]", mod, 1)

            # Recursively replace any remaining [R] in the new group
            if "[R]" in modified_smiles:
                modified_smiles = self.modify_r_groups(modified_smiles, depth + 1)

            if self.is_valid_smiles(modified_smiles):
                return modified_smiles  # Return first valid structure

        return None  # Return None if no valid structure is found

    def generate_multiple_smiles(self, num_samples=10):
        """Generate multiple SMILES strings with recursive modifications for [R]."""
        smiles_list = []
        for _ in range(num_samples):
            smiles = self.modify_r_groups()
            if smiles:
                smiles_list.append(smiles)
        return smiles_list
