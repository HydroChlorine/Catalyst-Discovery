from rdkit import Chem
from rdkit.Chem import rdchem
import random


class RecursiveSMILESGenerator:
    def __init__(self, base_smiles=r"[*]/[N+](N[*])=C\C1=CC=CC=C1", max_depth=3):
        self.base_smiles = base_smiles
        self.max_depth = max_depth

        # Fragments that introduce new branches while respecting valence
        self.functional_groups = [
            "C([*])",  # Add a methyl group (1 new [*])
            "C([*])([*])",  # Add a branching carbon (2 new [*])
            "N([*])",  # Secondary amine (safe for R-groups)
            "O",  # Ether (no new [*], terminates branch)
            "Cl",  # Halogen (terminates branch)
        ]

    def _tag_core_atoms(self, mol):
        """Tag atoms in the hydrazine core to ensure substitutions only occur at wildcards."""
        core_atoms = set()
        for atom in mol.GetAtoms():
            # Tag the [N+] and its directly bonded N as part of the core
            if atom.GetSymbol() == "N" and atom.GetFormalCharge() == 1:
                core_atoms.add(atom.GetIdx())
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == "N":
                        core_atoms.add(neighbor.GetIdx())
        return core_atoms

    def _replace_substituents(self, mol, core_atoms, depth=0):
        """Replace wildcards attached to the hydrazine core."""
        if depth > self.max_depth:
            return mol

        # Find wildcards attached to core atoms (these are the intended substitution points)
        substituent_wildcards = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbor = atom.GetNeighbors()[0]
                if neighbor.GetIdx() in core_atoms:  # <-- KEY CHANGE: Target core-attached [*]
                    substituent_wildcards.append(atom)

        if not substituent_wildcards:
            return mol

        # Replace a random substituent wildcard
        target = random.choice(substituent_wildcards)
        replacement = random.choice(self.functional_groups)
        frag = Chem.MolFromSmiles(replacement)
        if not frag:
            return None

        # Perform substitution
        combo = Chem.ReplaceSubstructs(
            mol,
            Chem.MolFromSmiles("[*]"),
            frag,
            replacementConnectionPoint=0
        )
        if not combo:
            return None
        new_mol = combo[0]

        # Recurse to process new wildcards (if any)
        return self._replace_substituents(new_mol, core_atoms, depth + 1)

    def generate_smiles(self):
        base_mol = Chem.MolFromSmiles(self.base_smiles)
        if not base_mol:
            return None

        core_atoms = self._tag_core_atoms(base_mol)
        final_mol = self._replace_substituents(base_mol, core_atoms)
        if not final_mol:
            return None

        # Cap remaining wildcards with [H]
        final_mol = Chem.ReplaceSubstructs(
            final_mol,
            Chem.MolFromSmiles("[*]"),
            Chem.MolFromSmiles("[H]"),
            replaceAll=True
        )[0]

        # Validate and return
        try:
            Chem.SanitizeMol(final_mol)
            return Chem.MolToSmiles(final_mol, canonical=True)
        except:
            return None

    def generate_multiple_smiles(self, num_samples=10):
        smiles_list = []
        for _ in range(num_samples * 2):
            if len(smiles_list) >= num_samples:
                break
            smiles = self.generate_smiles()
            if smiles and "[*]" not in smiles:
                smiles_list.append(smiles)
        return smiles_list[:num_samples]