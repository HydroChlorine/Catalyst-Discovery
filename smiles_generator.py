from rdkit import Chem
from rdkit.Chem import rdchem
import random
import numpy as np


class RecursiveSMILESGenerator:
    def __init__(self, base_smiles=r"[*]/[N+](N[*])=C\C1=CC=CC=C1", max_depth=6):
        self.base_smiles = base_smiles
        self.max_depth = max_depth

        # Multi-branching fragments with different probabilities
        self.fragments = {
            # High branching potential (weight 3.0)
            "C([*])([*])[*]": 3.0,  # 3 branches
            "N([*])([*])[*]": 2.0,  # 3 branches
            "C1([*])CC([*])CC1": 2.5,  # Cyclic with 2 branches

            # Moderate branching (weight 1.5)
            "C([*])=C([*])": 1.5,  # Double bond branch
            "C#C[*]": 1.5,  # Triple bond branch

            # Terminating groups (weight 0.5)
            "Cl": 0.5,
            "O": 0.5,
            "C": 0.3  # Keep methyl rare to avoid repetition
        }

    def _get_random_fragment(self, depth):
        """Depth-aware fragment selection (more branching early)"""
        weights = []
        for frag, base_weight in self.fragments.items():
            if depth < self.max_depth // 2:
                # Boost branching fragments early
                weight = base_weight * (1 + frag.count('[*]'))
            else:
                # Favor terminating groups late
                weight = base_weight / (1 + frag.count('[*]'))
            weights.append(weight)

        return random.choices(
            list(self.fragments.keys()),
            weights=weights,
            k=1
        )[0]

    def _recursive_replace(self, mol, depth=0):
        if depth >= self.max_depth:
            return mol

        wildcards = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "*"]
        if not wildcards:
            return mol

        # Replace ALL wildcards in random order
        random.shuffle(wildcards)
        for atom in wildcards.copy():
            try:
                replacement = self._get_random_fragment(depth)
                frag = Chem.MolFromSmiles(replacement)
                combo = Chem.ReplaceSubstructs(mol, Chem.MolFromSmiles("[*]"), frag)
                mol = combo[0]
                Chem.SanitizeMol(mol)
                # Immediately recurse for new wildcards
                mol = self._recursive_replace(mol, depth + 1)
            except:
                continue

        return mol

    def generate_smiles(self):
        base_mol = Chem.MolFromSmiles(self.base_smiles)
        if not base_mol:
            return None

        # Perform aggressive replacement
        final_mol = self._recursive_replace(base_mol)

        # Final cleanup: replace remaining * with random groups
        final_mol = Chem.ReplaceSubstructs(
            final_mol,
            Chem.MolFromSmiles("[*]"),
            Chem.MolFromSmiles(random.choice(["C", "Cl", "O"])),
            replaceAll=True
        )[0]

        try:
            Chem.SanitizeMol(final_mol)
            return Chem.MolToSmiles(final_mol, canonical=True)
        except:
            return None

    def generate_multiple_smiles(self, num_samples=10):
        seen = set()
        while len(seen) < num_samples:
            smiles = self.generate_smiles()
            if smiles and smiles not in seen:
                seen.add(smiles)
        return list(seen)