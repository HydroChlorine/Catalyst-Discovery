from rdkit import Chem
from rdkit.Chem import AllChem, rdchem, rdmolops
from rdkit.Chem.rdchem import BondType
from rdkit.Geometry import Point3D
import os
import argparse
import numpy as np


def smiles_to_gaussian_com(smiles, output_file="gibbs_calc.com", charge=0, mult=1,
                           mem="30GB", nproc=16, method="m062x", basis="6-31g(d)"):
    """
    Convert a SMILES string to a Gaussian .com file for Gibbs free energy calculation

    Args:
        smiles (str): Input SMILES string
        output_file (str): Output .com filename
        charge (int): Molecular charge
        mult (int): Spin multiplicity
        mem (str): Memory allocation
        nproc (int): Number of processors
        method (str): DFT method
        basis (str): Basis set
    """

    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Add hydrogens and generate 3D structure
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
    AllChem.MMFFOptimizeMolecule(mol)

    # Get atom coordinates
    conf = mol.GetConformer()
    coordinates = []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        coordinates.append(f"{atom.GetSymbol()} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")

    # Generate Gaussian input content
    content = f"""%mem={mem}
%nprocshared={nproc}
%chk={os.path.splitext(output_file)[0]}.chk
# {method}/{basis} freq=noraman int=grid=ultrafine temperature=298

{smiles} - Gibbs Free Energy Calculation

{charge} {mult}
""" + "\n".join(coordinates) + "\n\n"

    # Write to file
    with open(output_file, "w") as f:
        f.write(content)

    return output_file


def generate_ts_com(smiles, output_file="ts_calc.com", charge=1, mult=1,
                    mem="30GB", nproc=16, method="m062x", basis="6-31g(d)"):
    """
    Generate Gaussian input for transition state of hydrazine derivative + dec-5-ene [3+2] cycloaddition
    with specific hydrazine core: R¹-N⁺(R²)-N(R³)=C(H)-Ph

    Args:
        smiles (str): SMILES of hydrazine derivative
        output_file (str): Output .com filename
        charge (int): Molecular charge (default +1 for hydrazinium)
        mult (int): Spin multiplicity
        mem (str): Memory allocation
        nproc (int): Number of processors
        method (str): DFT method
        basis (str): Basis set
    """

    # 1. Load and validate hydrazine derivative
    hydrazine_mol = Chem.MolFromSmiles(smiles)
    if not hydrazine_mol:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Add hydrogens and generate 3D structure
    hydrazine_mol = Chem.AddHs(hydrazine_mol)
    AllChem.EmbedMolecule(hydrazine_mol, randomSeed=0xf00d)
    AllChem.MMFFOptimizeMolecule(hydrazine_mol)

    # 2. Identify key atoms in hydrazine core
    # Find the positively charged nitrogen (N⁺)
    n_plus_candidates = [atom for atom in hydrazine_mol.GetAtoms()
                         if atom.GetSymbol() == "N" and atom.GetFormalCharge() == 1]

    if not n_plus_candidates:
        raise ValueError("No positively charged nitrogen found in molecule")

    # Select the candidate with a double bond to carbon and single bond to nitrogen
    n_plus = None
    terminal_n = None
    iminium_c = None

    for candidate in n_plus_candidates:
        # Check neighbors
        neighbors = candidate.GetNeighbors()
        n_nbrs = 0
        c_nbrs = 0

        for nbr in neighbors:
            bond = hydrazine_mol.GetBondBetweenAtoms(candidate.GetIdx(), nbr.GetIdx())
            if nbr.GetSymbol() == "N" and bond.GetBondType() == BondType.SINGLE:
                terminal_n = nbr
                n_nbrs += 1
            elif nbr.GetSymbol() == "C" and bond.GetBondType() == BondType.DOUBLE:
                iminium_c = nbr
                c_nbrs += 1

        if n_nbrs == 1 and c_nbrs == 1:
            n_plus = candidate
            break

    if not n_plus:
        raise ValueError("Hydrazine core not found: N⁺ must have one single bond to N and one double bond to C")

    # 3. Verify terminal nitrogen has at least one hydrogen
    terminal_n_hs = [nbr for nbr in terminal_n.GetNeighbors()
                     if nbr.GetSymbol() == "H"]
    if not terminal_n_hs:
        # Check for implicit hydrogens
        if terminal_n.GetTotalNumHs() < 1:
            raise ValueError("Terminal nitrogen must have at least one hydrogen")

    # 4. Verify iminium carbon is attached to benzene
    benzene_ring = False
    for nbr in iminium_c.GetNeighbors():
        if nbr.GetSymbol() == "C" and nbr.GetIsAromatic():
            benzene_ring = True
            break

    if not benzene_ring:
        raise ValueError("Iminium carbon must be attached to a benzene ring")

    idx_N_plus = n_plus.GetIdx()
    idx_N_term = terminal_n.GetIdx()
    idx_C_iminium = iminium_c.GetIdx()

    # 5. Create dec-5-ene molecule
    decene_mol = Chem.MolFromSmiles("CCCCC=CCCCC")
    decene_mol = Chem.AddHs(decene_mol)
    AllChem.EmbedMolecule(decene_mol, randomSeed=0xf00d)
    AllChem.MMFFOptimizeMolecule(decene_mol)

    # Find the double bond in dec-5-ene
    double_bond = next((bond for bond in decene_mol.GetBonds()
                        if bond.GetBondType() == BondType.DOUBLE), None)
    if not double_bond:
        raise ValueError("No double bond found in dec-5-ene")

    idx_alkene_C1 = double_bond.GetBeginAtomIdx()
    idx_alkene_C2 = double_bond.GetEndAtomIdx()

    # 6. Create supermolecule (reactant complex)
    combined = Chem.CombineMols(hydrazine_mol, decene_mol)
    rw_combined = Chem.RWMol(combined)

    # 7. Align molecules for reaction
    hydrazine_conf = hydrazine_mol.GetConformer()
    decene_conf = decene_mol.GetConformer()

    # Get positions of key atoms
    pos_N_term = np.array(hydrazine_conf.GetAtomPosition(idx_N_term))
    pos_C_iminium = np.array(hydrazine_conf.GetAtomPosition(idx_C_iminium))
    pos_alkene_C1 = np.array(decene_conf.GetAtomPosition(idx_alkene_C1))
    pos_alkene_C2 = np.array(decene_conf.GetAtomPosition(idx_alkene_C2))

    # Calculate centroid of reactive atoms
    hydrazine_centroid = (pos_N_term + pos_C_iminium) / 2
    alkene_centroid = (pos_alkene_C1 + pos_alkene_C2) / 2

    # Calculate translation vector
    translation = hydrazine_centroid - alkene_centroid + np.array([3.0, 0.0, 0.0])

    # Create new conformer for combined system
    combined_conf = Chem.Conformer(combined.GetNumAtoms())

    # Copy hydrazine coordinates
    for i in range(hydrazine_mol.GetNumAtoms()):
        pos = hydrazine_conf.GetAtomPosition(i)
        combined_conf.SetAtomPosition(i, pos)

    # Copy and translate decene coordinates
    n_hydrazine = hydrazine_mol.GetNumAtoms()
    for i in range(decene_mol.GetNumAtoms()):
        j = i + n_hydrazine
        pos = decene_conf.GetAtomPosition(i)
        new_pos = Point3D(pos.x + translation[0],
                                 pos.y + translation[1],
                                 pos.z + translation[2])
        combined_conf.SetAtomPosition(j, new_pos)

    combined.AddConformer(combined_conf)

    # 8. Generate TS guess by forming partial bonds
    idx_N_term_combined = idx_N_term
    idx_C_iminium_combined = idx_C_iminium
    idx_alkene_C1_combined = idx_alkene_C1 + n_hydrazine
    idx_alkene_C2_combined = idx_alkene_C2 + n_hydrazine

    # Add partial bonds
    rw_combined.AddBond(idx_N_term_combined, idx_alkene_C1_combined, BondType.SINGLE)
    rw_combined.AddBond(idx_C_iminium_combined, idx_alkene_C2_combined, BondType.SINGLE)

    # Set bond lengths for TS guess
    ts_mol = rw_combined.GetMol()
    conf = ts_mol.GetConformer()
    target_distance = 2.0  # Angstrom

    # Move alkene C1 closer to terminal N
    vec_to_N_term = np.array(conf.GetAtomPosition(idx_N_term_combined)) - np.array(
        conf.GetAtomPosition(idx_alkene_C1_combined))
    current_distance = np.linalg.norm(vec_to_N_term)
    scale = target_distance / current_distance
    new_pos = np.array(conf.GetAtomPosition(idx_alkene_C1_combined)) + vec_to_N_term * (1 - scale)
    conf.SetAtomPosition(idx_alkene_C1_combined, Point3D(*new_pos))

    # Move alkene C2 closer to iminium C
    vec_to_C_iminium = np.array(conf.GetAtomPosition(idx_C_iminium_combined)) - np.array(
        conf.GetAtomPosition(idx_alkene_C2_combined))
    current_distance = np.linalg.norm(vec_to_C_iminium)
    scale = target_distance / current_distance
    new_pos = np.array(conf.GetAtomPosition(idx_alkene_C2_combined)) + vec_to_C_iminium * (1 - scale)
    conf.SetAtomPosition(idx_alkene_C2_combined, Point3D(*new_pos))

    # 9. Generate Gaussian input with connectivity section
    atom_lines = []
    for i, atom in enumerate(ts_mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        atom_lines.append(f"{atom.GetSymbol():2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")

    # Use a set to track which bonds have been added
    added_bonds = set()
    connectivity_lines = []
    atom_bonds = [[] for _ in range(ts_mol.GetNumAtoms())]

    # Only consider bonds within each molecule
    # Bonds in hydrazine
    for bond in hydrazine_mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        bond_order = int(bond.GetBondTypeAsDouble())
        bond_key = tuple(sorted([atom1_idx, atom2_idx]))
        min_idx = min(atom1_idx, atom2_idx)
        max_idx = max(atom1_idx, atom2_idx)
        if bond_key not in added_bonds:
            atom_bonds[min_idx].append(f"{max_idx + 1} {bond_order}.0")
            added_bonds.add(bond_key)

    # Bonds in decene
    for bond in decene_mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx() + n_hydrazine
        atom2_idx = bond.GetEndAtomIdx() + n_hydrazine
        bond_order = int(bond.GetBondTypeAsDouble())
        bond_key = tuple(sorted([atom1_idx, atom2_idx]))
        min_idx = min(atom1_idx, atom2_idx)
        max_idx = max(atom1_idx, atom2_idx)
        if bond_key not in added_bonds:
            atom_bonds[min_idx].append(f"{max_idx + 1} {bond_order}.0")
            added_bonds.add(bond_key)

    # Create connectivity lines
    for i in range(ts_mol.GetNumAtoms()):
        line = f"{i + 1}"
        if atom_bonds[i]:
            line += " " + " ".join(atom_bonds[i])
        connectivity_lines.append(line)

    # Route section for TS optimization
    route = (
        f"# opt=(ts,calcfc,noeigen) freq=noraman {method}/{basis} "
        "geom=connectivity int=grid=ultrafine temperature=298"
    )

    input_content = f"""%mem={mem}
%nprocshared={nproc}
%chk={os.path.splitext(output_file)[0]}.chk
{route}

{smiles} + dec-5-ene TS calculation

{charge} {mult}
""" + "\n".join(atom_lines) + "\n\n" + "\n".join(connectivity_lines) + "\n\n"

    # 10. Write to file
    with open(output_file, "w") as f:
        f.write(input_content)

    return output_file

if __name__ == "__main__":
    # Your example SMILES that previously failed
    example_smiles = "C[N+](N)=CC1=CC=CC=C1"
    try:
        output_file = generate_ts_com(example_smiles, "ts_calculation.com")
        print(f"Successfully created: {output_file}")
    except Exception as e:
        print(f"Error: {str(e)}")