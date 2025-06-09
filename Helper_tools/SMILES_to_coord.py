from rdkit import Chem
from rdkit.Chem import AllChem, rdchem, rdmolops
from rdkit.Chem.rdchem import BondType
from rdkit.Geometry import Point3D
import os
import argparse
import numpy as np
import math


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
    with proper hydrogen removal from terminal nitrogen
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
    n_plus_candidates = [atom for atom in hydrazine_mol.GetAtoms()
                         if atom.GetSymbol() == "N" and atom.GetFormalCharge() == 1]

    if not n_plus_candidates:
        raise ValueError("No positively charged nitrogen found in molecule")

    n_plus = None
    terminal_n = None
    iminium_c = None

    for candidate in n_plus_candidates:
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
    if not terminal_n_hs and terminal_n.GetTotalNumHs() < 1:
        raise ValueError("Terminal nitrogen must have at least one hydrogen")

    # 4. Verify iminium carbon is attached to benzene
    benzene_ring = False
    for nbr in iminium_c.GetNeighbors():
        if nbr.GetSymbol() == "C" and nbr.GetIsAromatic():
            benzene_ring = True
            break

    if not benzene_ring:
        for nbr in iminium_c.GetNeighbors():
            if nbr.GetSymbol() == "C":
                for nbr2 in nbr.GetNeighbors():
                    if nbr2.GetIsAromatic():
                        benzene_ring = True
                        break
                if benzene_ring:
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

    # 6. Create product molecule with hydrogen removal
    # Combine molecules
    combined = Chem.CombineMols(hydrazine_mol, decene_mol)
    product_mol = Chem.RWMol(combined)

    # Get atom indices
    n_hydrazine = hydrazine_mol.GetNumAtoms()

    #Remove one hydrogen from terminal nitrogen
    terminal_n_atom = product_mol.GetAtomWithIdx(idx_N_term)
    h_to_remove = None
    for nbr in terminal_n_atom.GetNeighbors():
        if nbr.GetSymbol() == "H":
            h_to_remove = nbr
            break

    if h_to_remove:
        product_mol.RemoveAtom(h_to_remove.GetIdx())
        # Update indices if we remove an atom before the alkene carbons
        if h_to_remove.GetIdx() < n_hydrazine:
            n_hydrazine -= 1
            if idx_alkene_C1 >= n_hydrazine:
                idx_alkene_C1 -= 1
            if idx_alkene_C2 >= n_hydrazine:
                idx_alkene_C2 -= 1

    # Add new bonds for cycloaddition
    product_mol.AddBond(idx_N_term, idx_alkene_C1 + n_hydrazine, BondType.SINGLE)
    product_mol.AddBond(idx_C_iminium, idx_alkene_C2 + n_hydrazine, BondType.SINGLE)

    # Modify existing bonds to match product state:
    # 1. Break the double bond in hydrazine (N+=C) -> make it single
    bond_to_break = hydrazine_mol.GetBondBetweenAtoms(idx_N_plus, idx_C_iminium)
    if bond_to_break and bond_to_break.GetBondType() == BondType.DOUBLE:
        product_mol.RemoveBond(idx_N_plus, idx_C_iminium)
        product_mol.AddBond(idx_N_plus, idx_C_iminium, BondType.SINGLE)

    # 2. Break the double bond in decene -> make it single
    bond_to_break = decene_mol.GetBondBetweenAtoms(idx_alkene_C1, idx_alkene_C2)
    if bond_to_break and bond_to_break.GetBondType() == BondType.DOUBLE:
        product_mol.RemoveBond(idx_alkene_C1 + n_hydrazine, idx_alkene_C2 + n_hydrazine)
        product_mol.AddBond(idx_alkene_C1 + n_hydrazine, idx_alkene_C2 + n_hydrazine, BondType.SINGLE)

    # Generate 3D structure for product
    product_mol = product_mol.GetMol()
    AllChem.EmbedMolecule(product_mol, randomSeed=0xf00d)
    AllChem.MMFFOptimizeMolecule(product_mol)


    # 7. Create TS by elongating forming bonds and restoring original double bonds
    ts_mol = Chem.RWMol(product_mol)
    conf = ts_mol.GetConformer()

    # Get atom indices in product
    idx_N_term_combined = idx_N_term
    idx_C_iminium_combined = idx_C_iminium
    idx_alkene_C1_combined = idx_alkene_C1 + n_hydrazine
    idx_alkene_C2_combined = idx_alkene_C2 + n_hydrazine

    # Calculate current bond vectors
    pos_N = np.array(conf.GetAtomPosition(idx_N_term_combined))
    pos_C_iminium = np.array(conf.GetAtomPosition(idx_C_iminium_combined))
    pos_C1 = np.array(conf.GetAtomPosition(idx_alkene_C1_combined))
    pos_C2 = np.array(conf.GetAtomPosition(idx_alkene_C2_combined))

    vec_N_C1 = pos_C1 - pos_N
    vec_C_C2 = pos_C2 - pos_C_iminium

    # Elongate bonds to TS distance (2.2 Å)
    ts_distance = 2.2  # Optimal TS bond length

    if np.linalg.norm(vec_N_C1) > 1e-6:
        new_vec_N_C1 = vec_N_C1 * (ts_distance / np.linalg.norm(vec_N_C1))
        new_pos_C1 = pos_N + new_vec_N_C1
        conf.SetAtomPosition(idx_alkene_C1_combined, Point3D(*new_pos_C1))

    if np.linalg.norm(vec_C_C2) > 1e-6:
        new_vec_C_C2 = vec_C_C2 * (ts_distance / np.linalg.norm(vec_C_C2))
        new_pos_C2 = pos_C_iminium + new_vec_C_C2
        conf.SetAtomPosition(idx_alkene_C2_combined, Point3D(*new_pos_C2))

    # Break the two new bonds
    ts_mol.RemoveBond(idx_N_term, idx_alkene_C1 + n_hydrazine)
    ts_mol.RemoveBond(idx_C_iminium, idx_alkene_C2 + n_hydrazine)

    # 8. Restore original double bonds in the TS structure
    # For hydrazine: change N+-C bond back to double
    bond = ts_mol.GetBondBetweenAtoms(idx_N_plus, idx_C_iminium)
    if bond:
        ts_mol.RemoveBond(idx_N_plus, idx_C_iminium)
        ts_mol.AddBond(idx_N_plus, idx_C_iminium, BondType.DOUBLE)

    # For decene: change C=C bond back to double
    bond = ts_mol.GetBondBetweenAtoms(idx_alkene_C1_combined, idx_alkene_C2_combined)
    if bond:
        ts_mol.RemoveBond(idx_alkene_C1_combined, idx_alkene_C2_combined)
        ts_mol.AddBond(idx_alkene_C1_combined, idx_alkene_C2_combined, BondType.DOUBLE)

    # Create new hydrogen atom
    #print('a')
    # Create new hydrogen atom
    new_h = Chem.Atom('H')
    h_idx = ts_mol.AddAtom(new_h)
    ts_mol.AddBond(idx_N_term, h_idx, BondType.SINGLE)

    # Remove all existing conformers
    ts_mol.RemoveAllConformers()

    # Create new conformer with correct atom count
    new_conf = Chem.Conformer(ts_mol.GetNumAtoms())

    # Copy coordinates from product_mol
    old_conf = product_mol.GetConformer()
    for i in range(product_mol.GetNumAtoms()):
        pos = old_conf.GetAtomPosition(i)
        new_conf.SetAtomPosition(i, pos)

    # Get terminal nitrogen position
    pos_N = new_conf.GetAtomPosition(idx_N_term)
    pos_N = np.array([pos_N.x, pos_N.y, pos_N.z])

    # Get positions of bonded atoms (N+ and existing H)
    neighbors = []
    terminal_n_atom = ts_mol.GetAtomWithIdx(idx_N_term)
    for nbr in terminal_n_atom.GetNeighbors():
        if nbr.GetIdx() == h_idx:  # Skip the new hydrogen we're adding
            continue
        pos = new_conf.GetAtomPosition(nbr.GetIdx())
        neighbors.append(np.array([pos.x, pos.y, pos.z]))

    # Calculate vectors from nitrogen to bonded atoms
    vectors = [v - pos_N for v in neighbors]

    # Calculate tetrahedral position for new hydrogen
    if len(vectors) == 2:
        # Normalize vectors
        v1 = vectors[0] / np.linalg.norm(vectors[0])
        v2 = vectors[1] / np.linalg.norm(vectors[1])

        # Calculate bisector and perpendicular
        bisector = (v1 + v2) / np.linalg.norm(v1 + v2)
        perpendicular = np.cross(v1, v2)
        if np.linalg.norm(perpendicular) < 1e-4:
            # Fallback if vectors are colinear
            perpendicular = np.array([1.0, 0.0, 0.0])
        else:
            perpendicular = perpendicular / np.linalg.norm(perpendicular)

        # Calculate new hydrogen direction (tetrahedral position)
        new_h_dir = -bisector - perpendicular
        new_h_dir = new_h_dir / np.linalg.norm(new_h_dir)
    else:
        # Fallback position if unexpected number of neighbors
        new_h_dir = np.array([1.0, 0.0, 0.0])

    # Set new hydrogen position (1.0 Å from nitrogen)
    new_h_pos = pos_N + new_h_dir * 1.0
    new_conf.SetAtomPosition(h_idx, Point3D(*new_h_pos))

    # Add the new conformer to ts_mol
    conf_id = ts_mol.AddConformer(new_conf)
    conf = ts_mol.GetConformer(conf_id)

    # Update property cache
    ts_mol.UpdatePropertyCache(strict=False)

    # 9. Optimize hydrogen position using RDKit's local optimization
    try:
        mp = AllChem.MMFFGetMoleculeProperties(ts_mol)
        if mp:
            ff = AllChem.MMFFGetMoleculeForceField(
                ts_mol, mp, confId=conf_id, ignoreInterfragInteractions=True
            )
            if ff:
                # Freeze all atoms except the new hydrogen
                for i in range(ts_mol.GetNumAtoms()):
                    if i != h_idx:
                        ff.AddFixedPoint(i)
                ff.Minimize(maxIts=200)
    except:
        pass

    # 10. Calculate ring information before Kekulization
    ts_mol.UpdatePropertyCache(strict=False)
    Chem.GetSSSR(ts_mol)  # This initializes ring information
    Chem.Kekulize(ts_mol)

    # 10. Generate Gaussian input with connectivity section
    atom_lines = []
    for i, atom in enumerate(ts_mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        atom_lines.append(f"{atom.GetSymbol():2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")

    # Generate connectivity section without duplicates
    added_bonds = set()
    connectivity_lines = []
    atom_bonds = [[] for _ in range(ts_mol.GetNumAtoms())]

    # Iterate through all bonds
    for bond in ts_mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        bond_order = int(bond.GetBondTypeAsDouble())

        # Create canonical representation
        bond_key = tuple(sorted([atom1_idx, atom2_idx]))

        # Add bond to lower-indexed atom
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

    # 11. Write to file
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