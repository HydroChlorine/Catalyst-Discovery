from rdkit import Chem
from rdkit.Chem import rdchem, rdmolops
import random
import time
from collections import deque
from tqdm import tqdm


class RecursiveSMILESGenerator:
    def __init__(self,
                 base_smiles=r"[*]/[N+](N[*])=C\C1=CC=CC=C1",
                 max_depth=4,
                 ring_prob=0.3,
                 min_ring_size=5,
                 max_ring_size=6):

        self.base_smiles = base_smiles
        self.max_depth = max_depth
        self.ring_prob = ring_prob
        self.min_ring_size = min_ring_size
        self.max_ring_size = max_ring_size

        # Optimized fragment set with better termination balance
        self.fragments = [
            # Core fragments with weights
            ("C([*])", 0.5),  # Single branch
            ("C", 0.3),  # Termination
            ("Cl", 0.2), ("O", 0.2),  # Halogen/oxygen termination

            # New double bond fragments
            ("C([*])=C([*])", 0.3),  # Double bond with 2 branches
            ("C=C[*]", 0.2),  # Terminal double bond
            ("C(=C[*])[*]", 0.2),  # Branched double bond

            # New triple bond fragments
            ("C#C[*]", 0.2),  # Terminal triple bond
            ("C([*])#C([*])", 0.1),  # Internal triple bond

            # Nitrogen branching fragments
            ("N([*])[*]", 0.4),  # Secondary amine branch
            ("N([*])([*])", 0.3),  # Tertiary amine branch
            ("N(=O)[*]", 0.2),  # Nitro group branch

            # High-branching carbon fragments
            ("C([*])([*])[*]", 0.4),  # Tertiary carbon
            ("C([*])([*])([*])", 0.2),  # Quaternary carbon
            ("C1([*])CC([*])CC1", 0.3),  # Cyclohexane with branches

            # Oxygen-containing fragments
            ("O[*]", 0.4),  # Ether linkage
            ("OC([*])=O", 0.2),  # Carboxylic acid branch
            ("O=C([*])[*]", 0.3),  # Ketone branch

            # Sulfur-containing fragments
            ("S[*]", 0.2),  # Thioether
            ("S(=O)(=O)[*]", 0.1)  # Sulfonyl group
        ]

        self.wildcard = Chem.MolFromSmiles("[*]")
        self.base_mol = Chem.MolFromSmiles(base_smiles)
        if not self.base_mol:
            raise ValueError("Invalid base SMILES string")

    def _find_ring_partners(self, mol, start_wildcard):
        """Efficient BFS with strict cycle prevention"""
        partners = []
        start_parent = start_wildcard.GetNeighbors()[0]

        queue = deque([(start_parent, 0)])
        visited = set([start_parent.GetIdx()])

        while queue:
            current_atom, depth = queue.popleft()

            # Limit search depth to max_ring_size
            if depth > self.max_ring_size:
                continue

            for neighbor in current_atom.GetNeighbors():
                if neighbor.GetSymbol() == "*":
                    # Found potential partner
                    partner_parent = neighbor.GetNeighbors()[0]
                    if (partner_parent.GetIdx() != start_parent.GetIdx() and
                            not mol.GetBondBetweenAtoms(start_parent.GetIdx(), partner_parent.GetIdx())):
                        path_length = depth + 2
                        if self.min_ring_size <= path_length <= self.max_ring_size:
                            partners.append(neighbor)
                elif neighbor.GetIdx() not in visited:
                    visited.add(neighbor.GetIdx())
                    queue.append((neighbor, depth + 1))

        return random.sample(partners, min(3, len(partners))) if partners else []

    def _attempt_ring_formation(self, mol, wildcard):
        """Fast ring formation with atomic index protection"""
        partners = self._find_ring_partners(mol, wildcard)
        if not partners:
            return None

        for partner in partners:
            try:
                # Work on a copy of the molecule
                rw_mol = Chem.RWMol(mol)
                start_parent = wildcard.GetNeighbors()[0]
                partner_parent = partner.GetNeighbors()[0]

                # Create bond first to preserve indices
                rw_mol.AddBond(start_parent.GetIdx(), partner_parent.GetIdx(), Chem.BondType.SINGLE)

                # Remove wildcards by atom indices
                rw_mol.RemoveAtom(wildcard.GetIdx())
                rw_mol.RemoveAtom(partner.GetIdx())

                # Validate and return
                new_mol = rw_mol.GetMol()
                Chem.SanitizeMol(new_mol)
                return new_mol
            except Exception as e:
                continue
        return None

    def _safe_replace(self, mol):
        """Iterative replacement with guaranteed termination"""
        current_mol = mol
        depth = 0
        max_attempts = 50

        while depth < self.max_depth and max_attempts > 0:
            max_attempts -= 1
            wildcards = [atom for atom in current_mol.GetAtoms() if atom.GetSymbol() == "*"]

            if not wildcards:
                break

            # Attempt ring formation first
            if random.random() < self.ring_prob:
                target = random.choice(wildcards)
                ring_mol = self._attempt_ring_formation(current_mol, target)
                if ring_mol:
                    current_mol = ring_mol
                    depth += 1
                    continue

            # Regular fragment replacement
            frag = random.choices(
                [f for f, _ in self.fragments],
                weights=[w for _, w in self.fragments],
                k=1
            )[0]

            new_mol = Chem.ReplaceSubstructs(current_mol, self.wildcard,
                                             Chem.MolFromSmiles(frag))[0]
            if new_mol == current_mol:  # No substitution occurred
                break

            current_mol = new_mol
            depth += 1

        return current_mol

    def generate_smiles(self, max_attempts=100):
        """High-performance generation with attempt tracking"""
        for _ in range(max_attempts):
            try:
                # Start with fresh molecule each attempt
                base_mol = Chem.MolFromSmiles(self.base_smiles)
                result = self._safe_replace(base_mol)

                # Finalize remaining wildcards
                result = Chem.ReplaceSubstructs(
                    result,
                    self.wildcard,
                    Chem.MolFromSmiles("C"),
                    replaceAll=True
                )[0]

                Chem.SanitizeMol(result)
                smiles = Chem.MolToSmiles(result, canonical=True)
                if smiles and "*" not in smiles:
                    return smiles
            except:
                continue
        return None

    def generate_multiple_smiles(self, num_samples, timeout=60):
        """Reliable batch generation with progress tracking"""
        results = set()
        start_time = time.time()

        with tqdm(total=num_samples, desc="Generating") as pbar:
            while len(results) < num_samples and (time.time() - start_time) < timeout:
                smiles = self.generate_smiles()
                if smiles and smiles not in results:
                    results.add(smiles)
                    pbar.update(1)

        return list(results)