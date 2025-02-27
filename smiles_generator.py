from rdkit import Chem
from rdkit.Chem import rdchem
import random
import time
from tqdm import tqdm


class RecursiveSMILESGenerator:
    def __init__(self, base_smiles=r"[*]/[N+](N[*])=C\C1=CC=CC=C1", max_depth=3):
        self.base_smiles = base_smiles
        self.max_depth = max_depth

        # Updated fragments with nitrogen and oxygen groups
        self.fragments = [
            # Carbon-based
            ("C", 0.2),  # Simple termination
            ("C([*])", 0.4),  # Single branch
            ("C([*])([*])", 0.2),  # Double branch

            # Nitrogen-based (valence-safe)
            ("N([*])", 0.3),  # Secondary amine
            ("N([*])(C)", 0.2),  # Tertiary amine (methyl group)
            ("NC([*])", 0.2),  # Amino group with branch

            # Oxygen-based
            ("O[*]", 0.4),  # Ether linkage
            ("OC([*])", 0.3),  # Methoxy with branch
            ("O", 0.3),  # Hydroxyl termination

            # Halogens and others
            ("Cl", 0.3),
            ("F", 0.2)
        ]

        self.wildcard = Chem.MolFromSmiles("[*]")
        self.base_mol = Chem.MolFromSmiles(base_smiles)

        # Validate base molecule
        if not self.base_mol:
            raise ValueError("Invalid base SMILES string")

    def _safe_replace(self, mol, depth):
        """Non-recursive replacement with depth tracking"""
        try:
            temp = Chem.RWMol(mol)
            wildcards = [atom for atom in temp.GetAtoms() if atom.GetSymbol() == "*"]

            if not wildcards or depth > self.max_depth:
                return temp

            # Replace a random wildcard
            target = random.choice(wildcards)
            frag, _ = random.choices(
                self.fragments,
                weights=[w for _, w in self.fragments],
                k=1
            )[0]

            # Perform substitution
            combo = Chem.ReplaceSubstructs(
                temp.GetMol(),
                self.wildcard,
                Chem.MolFromSmiles(frag),
                replaceAll=False
            )[0]

            return self._safe_replace(combo, depth + 1)

        except:
            return None

    def generate_smiles(self, max_attempts=50):
        """Generate one valid SMILES with attempt limiting"""
        for _ in range(max_attempts):
            try:
                result = self._safe_replace(self.base_mol, 0)
                if not result:
                    continue

                # Finalize molecule
                result = Chem.ReplaceSubstructs(
                    result,
                    self.wildcard,
                    Chem.MolFromSmiles("C"),  # Final replacement with carbon
                    replaceAll=True
                )[0]

                Chem.SanitizeMol(result)
                return Chem.MolToSmiles(result, canonical=True)

            except Exception as e:
                continue
        return None

    def generate_multiple_smiles(self, num_samples, timeout=30):
        """Guaranteed return with timeout"""
        results = set()
        start = time.time()

        with tqdm(total=num_samples, desc="Generating") as pbar:
            while len(results) < num_samples and (time.time() - start) < timeout:
                smiles = self.generate_smiles()
                if smiles and smiles not in results:
                    results.add(smiles)
                    pbar.update(1)

        return list(results)