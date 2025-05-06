from rdkit import Chem
from rdkit.Chem import rdchem, rdmolops, GetPeriodicTable
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
        self.pt = GetPeriodicTable()
        self.target_ranges = [(),(),(),(),(),()]
        # Valence-safe fragments (SMILES, weight, max_bonds_dict)
        self.fragments = [
            ("C([*])", 0.6, {}),
            ("C", 0.4, {}),
            ("Cl", 0.3, {}),
            ("O", 0.3, {'O': 1}),
            ("C([*])=C([*])", 0.3, {'C': 4}),
            ("C#C[*]", 0.2, {'C': 4}),
            ("N([*])[*]", 0.4, {'N': 3}),
            ("N([*])([*])", 0.3, {'N': 3}),
            ("O[*]", 0.4, {'O': 2}),
            ("C(=O)(O[*])", 0.3, {'O' : 2}),
            ("C(=O)[*]", 0.4, {'O': 2, 'C': 4}),
            ("C1([*])CC([*])CC1", 0.3, {'C': 4}),
            ("S[*]", 0.2, {'S': 2}),

            # Terminal aromatic groups (no wildcards)
            ("C1=CC=CC=C1", 0.0, {}),  # Benzene terminal
            ("C1=CC=NC=C1", 0.2, {'N': 3}),  # Pyridine terminal
            ("C1=CSC=C1", 0.2, {'S': 2}),  # Thiophene terminal

            # Revised benzene fragments with explicit bond orders
            ("C1=C(C=CC(=C1)[*])", 0.5, {}),  # Para substitution
            ("C1=C(C(=C(C=C1)[*]))", 0.4, {}),  # Meta substitution
            ("C1=C([*])C=CC=C1", 0.3, {}),  # Ortho substitution

            # Di-substituted benzene derivatives
            ("C1=C([*])C(=CC=C1)[*]", 0.3, {}),  # Ortho-di
            ("C1=C([*])C=CC(=C1)[*]", 0.3, {}),  # Para-di
            ("C1=CC(=C(C=C1)[*])[*]", 0.3, {}),  # Meta-di

            # Tri-substituted benzene
            ("C1=C([*])C(=C([*])C=C1)[*]", 0.2, {}),  # 1,2,3-tri
            ("C1=C([*])C=CC(=C1[*])[*]", 0.2, {}),  # 1,2,4-tri

            # Fused aromatic systems
            ("C1=CC=C2C(=C1)C=CC(=C2)[*]", 0.3, {}),  # Naphthalene substitution
            ("C1=CC=C2C(=C1)C=C([*])C=C2", 0.2, {}),  # Naphthalene different position

            # Heteroaromatic systems
            ("C1=CC(=NC=C1)[*]", 0.4, {'N': 3}),  # Pyridine (meta)
            ("C1=C([*])SC=C1", 0.3, {'S': 2}),  # Thiophene
            ("C1=C([*])OC=C1", 0.3, {'O': 2}),  # Furan
            ("C1=C([*])N=C(C=C1)[*]", 0.2, {'N': 3}),  # Pyrimidine

            # Polyaromatic hydrocarbons
            ("C1=CC=C(C=C1)C2=CC=CC=C2[*]", 0.2, {}),  # Biphenyl
            ("C1=CC=CC(=C1)C2=C(C=CC=C2)[*]", 0.1, {}),  # Stilbene-like

        ]

        self.wildcard = Chem.MolFromSmiles("[*]")
        self.base_mol = Chem.MolFromSmiles(base_smiles)
        if not self.base_mol:
            raise ValueError("Invalid base SMILES string")

    def _find_ring_partners(self, mol, start_wildcard):
        """Find viable ring partners with BFS and depth control"""
        partners = []
        start_parent = start_wildcard.GetNeighbors()[0]

        queue = deque([(start_parent, 0)])
        visited = set([start_parent.GetIdx()])

        while queue:
            current_atom, depth = queue.popleft()

            if depth > self.max_ring_size:
                continue

            for neighbor in current_atom.GetNeighbors():
                if neighbor.GetSymbol() == "*":
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

    def _validate_valence(self, atom, max_bonds):
        """Check valence against periodic table and custom limits"""
        symbol = atom.GetSymbol()
        default = self.pt.GetDefaultValence(symbol)
        allowed = max_bonds.get(symbol, default)
        return atom.GetExplicitValence() <= allowed

    def _attempt_ring_formation(self, mol, wildcard):
        """Safe ring formation with valence checks"""
        partners = self._find_ring_partners(mol, wildcard)
        if not partners:
            return None

        for partner in partners:
            try:
                rw_mol = Chem.RWMol(mol)
                start_parent = wildcard.GetNeighbors()[0]
                partner_parent = partner.GetNeighbors()[0]

                sp_idx = start_parent.GetIdx()
                pp_idx = partner_parent.GetIdx()

                if rw_mol.GetBondBetweenAtoms(sp_idx, pp_idx):
                    continue

                rw_mol.AddBond(sp_idx, pp_idx, Chem.BondType.SINGLE)

                # Validate valence after bond addition
                if not (self._validate_valence(rw_mol.GetAtomWithIdx(sp_idx), {}) and
                        self._validate_valence(rw_mol.GetAtomWithIdx(pp_idx), {})):
                    continue

                # Remove wildcards safely
                for idx in sorted([wildcard.GetIdx(), partner.GetIdx()], reverse=True):
                    if idx < rw_mol.GetNumAtoms():
                        rw_mol.RemoveAtom(idx)

                new_mol = rw_mol.GetMol()
                Chem.SanitizeMol(new_mol)
                return new_mol
            except:
                continue
        return None

    def _replace_fragment(self, mol, depth):
        """Valence-aware fragment replacement"""
        wildcards = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "*"]
        if not wildcards or depth >= self.max_depth:
            return mol

        target = random.choice(wildcards)
        parent = target.GetNeighbors()[0]

        valid_frags = []
        for frag, weight, max_bonds in self.fragments:
            frag_mol = Chem.MolFromSmiles(frag)
            if not frag_mol:
                continue

            connection = frag_mol.GetAtomWithIdx(0)
            new_valence = parent.GetExplicitValence() + connection.GetExplicitValence()

            if new_valence <= self.pt.GetDefaultValence(parent.GetSymbol()):
                valid_frags.append((frag, weight))

        if not valid_frags:
            return mol

        frag = random.choices(
            [f for f, _ in valid_frags],
            weights=[w for _, w in valid_frags],
            k=1
        )[0]

        return Chem.ReplaceSubstructs(mol, self.wildcard,
                                      Chem.MolFromSmiles(frag))[0]

    def _safe_replace(self, mol):
        """Iterative replacement engine"""
        current_mol = mol
        depth = 0
        max_attempts = 50

        while depth < self.max_depth and max_attempts > 0:
            max_attempts -= 1

            # Attempt ring formation
            if random.random() < self.ring_prob:
                wildcards = [a for a in current_mol.GetAtoms() if a.GetSymbol() == "*"]
                if wildcards:
                    target = random.choice(wildcards)
                    ring_mol = self._attempt_ring_formation(current_mol, target)
                    if ring_mol:
                        current_mol = ring_mol
                        depth += 1
                        continue

            # Fragment replacement
            new_mol = self._replace_fragment(current_mol, depth)
            if new_mol == current_mol:
                break

            current_mol = new_mol
            depth += 1

        return current_mol

    def generate_smiles(self, max_attempts=100):
        """Generate single valid SMILES"""
        for _ in range(max_attempts):
            try:
                base_mol = Chem.MolFromSmiles(self.base_smiles)
                result = self._safe_replace(base_mol)

                # Finalize wildcards
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
        """Batch generation with progress tracking"""
        results = set()
        start_time = time.time()

        with tqdm(total=num_samples, desc="Generating") as pbar:
            while len(results) < num_samples and (time.time() - start_time) < timeout:
                smiles = self.generate_smiles()
                if smiles and smiles not in results:
                    results.add(smiles)
                    pbar.update(1)

        return list(results)