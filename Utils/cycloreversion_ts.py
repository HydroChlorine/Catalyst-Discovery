from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType
from rdkit.Geometry import Point3D
import os
import numpy as np
from Utils.find_all_reachable_neighbors import find_all_reachable_neighbors


def generate_proton_transfer_product_e4(smiles, output_file="proton_transfer_product_e4.com", charge=1, mult=1,
                                        mem="180GB", nproc=40, method="m062x", basis="6-311+g(2d,p)"):
    """
    Generate Gaussian input for proton-transferred product of cycloaddition (stable ion)

    Args:
        smiles (str): SMILES for hydrazine derivative
        output_file (str): Output filename
        charge (int): Molecular charge
        mult (int): Spin multiplicity
        mem (str): Memory allocation
        nproc (int): Number of processors
        method (str): DFT method
        basis (str): Basis set
    """
    # 1. Build the cycloaddition product structure
    hydrazine_mol = Chem.MolFromSmiles(smiles)
    if not hydrazine_mol:
        raise ValueError(f"Invalid SMILES: {smiles}")

    hydrazine_mol = Chem.AddHs(hydrazine_mol)
    AllChem.EmbedMolecule(hydrazine_mol, randomSeed=0xf00d)
    AllChem.MMFFOptimizeMolecule(hydrazine_mol)

    # Identify key atoms in hydrazine core
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

    terminal_n_hs = [nbr for nbr in terminal_n.GetNeighbors()
                     if nbr.GetSymbol() == "H"]
    if not terminal_n_hs and terminal_n.GetTotalNumHs() < 1:
        raise ValueError("Terminal nitrogen must have at least one hydrogen")

    # Verify iminium carbon is attached to benzene
    benzene_ring = False
    for nbr in iminium_c.GetNeighbors():
        if nbr.GetSymbol() == "C" and nbr.GetIsAromatic():
            benzene_ring = True
            break

    if not benzene_ring:
        raise ValueError("Iminium carbon must be attached to a benzene ring")

    # Create dec-5-ene molecule
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

    # Create product molecule
    combined = Chem.CombineMols(hydrazine_mol, decene_mol)
    product_mol = Chem.RWMol(combined)
    n_hydrazine = hydrazine_mol.GetNumAtoms()

    # Add new bonds for cycloaddition
    product_mol.AddBond(terminal_n.GetIdx(), idx_alkene_C1 + n_hydrazine, BondType.SINGLE)
    product_mol.AddBond(iminium_c.GetIdx(), idx_alkene_C2 + n_hydrazine, BondType.SINGLE)

    # Modify existing bonds
    bond_to_modify = product_mol.GetBondBetweenAtoms(n_plus.GetIdx(), iminium_c.GetIdx())
    if bond_to_modify and bond_to_modify.GetBondType() == BondType.DOUBLE:
        product_mol.RemoveBond(n_plus.GetIdx(), iminium_c.GetIdx())
        product_mol.AddBond(n_plus.GetIdx(), iminium_c.GetIdx(), BondType.SINGLE)

    bond_to_modify = product_mol.GetBondBetweenAtoms(
        idx_alkene_C1 + n_hydrazine, idx_alkene_C2 + n_hydrazine
    )
    if bond_to_modify and bond_to_modify.GetBondType() == BondType.DOUBLE:
        product_mol.RemoveBond(
            idx_alkene_C1 + n_hydrazine, idx_alkene_C2 + n_hydrazine
        )
        product_mol.AddBond(
            idx_alkene_C1 + n_hydrazine, idx_alkene_C2 + n_hydrazine, BondType.SINGLE
        )

    # Adjust formal charges for cycloadduct
    n_plus_atom = product_mol.GetAtomWithIdx(n_plus.GetIdx())
    n_plus_atom.SetFormalCharge(0)
    terminal_n_atom = product_mol.GetAtomWithIdx(terminal_n.GetIdx())
    terminal_n_atom.SetFormalCharge(1)

    # Generate and optimize 3D structure for cycloadduct
    product_mol = product_mol.GetMol()
    product_mol.UpdatePropertyCache(strict=False)
    AllChem.EmbedMolecule(product_mol, randomSeed=0xf00d)
    AllChem.MMFFOptimizeMolecule(product_mol)

    # 2. Perform proton transfer to create stable ion
    ion_mol = Chem.RWMol(product_mol)
    conf = ion_mol.GetConformer()

    # Identify the hydrogen to transfer from terminal nitrogen
    h_to_transfer = None
    terminal_n_idx = terminal_n.GetIdx()
    for nbr in ion_mol.GetAtomWithIdx(terminal_n_idx).GetNeighbors():
        if nbr.GetSymbol() == "H":
            h_to_transfer = nbr
            break

    if not h_to_transfer:
        raise ValueError("No hydrogen found on terminal nitrogen for proton transfer")

    # Remove hydrogen from terminal nitrogen
    h_idx = h_to_transfer.GetIdx()
    ion_mol.RemoveAtom(h_idx)

    # Adjust indices after removal
    if h_idx < n_hydrazine:
        n_hydrazine -= 1
        if idx_alkene_C1 >= n_hydrazine:
            idx_alkene_C1 -= 1
        if idx_alkene_C2 >= n_hydrazine:
            idx_alkene_C2 -= 1
        terminal_n_idx = terminal_n.GetIdx()
        if terminal_n_idx >= h_idx:
            terminal_n_idx -= 1
        n_plus_idx = n_plus.GetIdx()
        if n_plus_idx >= h_idx:
            n_plus_idx -= 1
        iminium_c_idx = iminium_c.GetIdx()
        if iminium_c_idx >= h_idx:
            iminium_c_idx -= 1

    # Add hydrogen to middle nitrogen
    new_h = Chem.Atom('H')
    new_h_idx = ion_mol.AddAtom(new_h)
    ion_mol.AddBond(n_plus_idx, new_h_idx, BondType.SINGLE)

    # Adjust charges for proton transfer
    ion_mol.GetAtomWithIdx(terminal_n_idx).SetFormalCharge(0)  # Terminal N becomes neutral
    ion_mol.GetAtomWithIdx(n_plus_idx).SetFormalCharge(1)  # Middle N becomes positive

    # 3. Generate and optimize 3D structure for the ion
    ion_mol = ion_mol.GetMol()
    ion_mol.UpdatePropertyCache(strict=False)

    # Create new conformer with correct atom count
    new_conf = Chem.Conformer(ion_mol.GetNumAtoms())
    old_conf = product_mol.GetConformer()

    # Copy coordinates from product_mol
    for i in range(product_mol.GetNumAtoms()):
        if i < h_idx:
            pos = old_conf.GetAtomPosition(i)
            new_conf.SetAtomPosition(i, pos)
        elif i > h_idx:
            pos = old_conf.GetAtomPosition(i)
            new_conf.SetAtomPosition(i - 1, pos)

    # Set initial position for new H near middle nitrogen
    pos_N_plus = old_conf.GetAtomPosition(n_plus.GetIdx())
    new_conf.SetAtomPosition(new_h_idx, Point3D(pos_N_plus.x, pos_N_plus.y, pos_N_plus.z + 1.0))

    # Add the new conformer to ion_mol
    ion_mol.AddConformer(new_conf)

    # Optimize the entire structure
    AllChem.MMFFOptimizeMolecule(ion_mol)

    # 4. Generate Gaussian input content with connectivity section
    conf = ion_mol.GetConformer()
    atom_lines = []
    for i, atom in enumerate(ion_mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        atom_lines.append(f"{atom.GetSymbol():2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")

    # Generate connectivity section
    added_bonds = set()
    connectivity_lines = []
    atom_bonds = [[] for _ in range(ion_mol.GetNumAtoms())]

    for bond in ion_mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        bond_order = bond.GetBondTypeAsDouble()

        # Create canonical representation
        bond_key = tuple(sorted([atom1_idx, atom2_idx]))

        # Add bond to lower-indexed atom
        min_idx = min(atom1_idx, atom2_idx)
        max_idx = max(atom1_idx, atom2_idx)

        if bond_key not in added_bonds:
            atom_bonds[min_idx].append(f"{max_idx + 1} {bond_order}")
            added_bonds.add(bond_key)

    for i in range(ion_mol.GetNumAtoms()):
        line = f"{i + 1}"
        if atom_bonds[i]:
            line += " " + " ".join(atom_bonds[i])
        connectivity_lines.append(line)

    # Route section for optimization and frequency calculation
    route = (
        f"# {basis} scrf=(solvent=Dichloromethane,CPCM) geom=connectivity\n"
        f"empiricaldispersion=gd3 int=grid=ultrafine {method} sp"
    )

    input_content = f"""%mem={mem}
%nprocshared={nproc}
%chk={os.path.splitext(output_file)[0]}.chk
{route}

{smiles} + dec-5-ene proton-transferred product - Gibbs free energy calculation

{charge} {mult}
""" + "\n".join(atom_lines) + "\n\n" + "\n".join(connectivity_lines) + "\n\n"

    # 5. Write to file
    with open(output_file, "w") as f:
        f.write(input_content)

    return output_file


def generate_cycloreversion_ts_e5(smiles, output_file="ts_cycloreversion_e5.com", charge=1, mult=1,
                                  mem="180GB", nproc=40, method="m062x", basis="6-31g(d)"):
    """
    Generate Gaussian input for cycloreversion TS (E5 step) of hydrazine derivative + dec-5-ene
    """
    # 1. Build the cycloaddition product with proton transfer
    # Convert SMILES to RDKit molecule
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

    terminal_n_hs = [nbr for nbr in terminal_n.GetNeighbors()
                     if nbr.GetSymbol() == "H"]
    if not terminal_n_hs and terminal_n.GetTotalNumHs() < 1:
        raise ValueError("Terminal nitrogen must have at least one hydrogen")

    # Verify iminium carbon is attached to benzene
    benzene_ring = False
    for nbr in iminium_c.GetNeighbors():
        if nbr.GetSymbol() == "C" and nbr.GetIsAromatic():
            benzene_ring = True
            break

    if not benzene_ring:
        raise ValueError("Iminium carbon must be attached to a benzene ring")

    # Record key indices
    idx_N_plus = n_plus.GetIdx()
    idx_N_term = terminal_n.GetIdx()
    idx_C_iminium = iminium_c.GetIdx()

    # 3. Create dec-5-ene molecule
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

    # 4. Create combined molecule
    combined = Chem.CombineMols(hydrazine_mol, decene_mol)
    product_mol = Chem.RWMol(combined)
    n_hydrazine = hydrazine_mol.GetNumAtoms()

    # 5. Add new bonds for cycloaddition
    product_mol.AddBond(idx_N_term, idx_alkene_C1 + n_hydrazine, BondType.SINGLE)
    product_mol.AddBond(idx_C_iminium, idx_alkene_C2 + n_hydrazine, BondType.SINGLE)

    # 6. Modify existing bonds
    bond_to_modify = product_mol.GetBondBetweenAtoms(idx_N_plus, idx_C_iminium)
    if bond_to_modify and bond_to_modify.GetBondType() == BondType.DOUBLE:
        product_mol.RemoveBond(idx_N_plus, idx_C_iminium)
        product_mol.AddBond(idx_N_plus, idx_C_iminium, BondType.SINGLE)

    bond_to_modify = product_mol.GetBondBetweenAtoms(
        idx_alkene_C1 + n_hydrazine, idx_alkene_C2 + n_hydrazine
    )
    if bond_to_modify and bond_to_modify.GetBondType() == BondType.DOUBLE:
        product_mol.RemoveBond(
            idx_alkene_C1 + n_hydrazine, idx_alkene_C2 + n_hydrazine
        )
        product_mol.AddBond(
            idx_alkene_C1 + n_hydrazine, idx_alkene_C2 + n_hydrazine, BondType.SINGLE
        )

    # 7. Adjust formal charges for cycloadduct
    product_mol.GetAtomWithIdx(idx_N_plus).SetFormalCharge(0)
    product_mol.GetAtomWithIdx(idx_N_term).SetFormalCharge(1)

    # 8. Perform proton transfer
    # Identify a hydrogen on terminal nitrogen to transfer
    terminal_n_atom = product_mol.GetAtomWithIdx(idx_N_term)
    h_to_transfer = None
    for nbr in terminal_n_atom.GetNeighbors():
        if nbr.GetSymbol() == "H":
            h_to_transfer = nbr
            break

    if not h_to_transfer:
        raise ValueError("No hydrogen found on terminal nitrogen for proton transfer")

    # Remove hydrogen from terminal nitrogen
    h_idx = h_to_transfer.GetIdx()
    product_mol.RemoveAtom(h_idx)

    # Adjust indices after removal
    if h_idx < n_hydrazine:
        n_hydrazine -= 1
        if idx_alkene_C1 >= n_hydrazine:
            idx_alkene_C1 -= 1
        if idx_alkene_C2 >= n_hydrazine:
            idx_alkene_C2 -= 1
        if idx_N_term >= h_idx:
            idx_N_term -= 1
        if idx_N_plus >= h_idx:
            idx_N_plus -= 1
        if idx_C_iminium >= h_idx:
            idx_C_iminium -= 1

    # Add hydrogen to middle nitrogen
    new_h = Chem.Atom('H')
    new_h_idx = product_mol.AddAtom(new_h)
    product_mol.AddBond(idx_N_plus, new_h_idx, BondType.SINGLE)

    # Adjust charges for proton transfer
    product_mol.GetAtomWithIdx(idx_N_term).SetFormalCharge(0)  # Terminal N becomes neutral
    product_mol.GetAtomWithIdx(idx_N_plus).SetFormalCharge(1)  # Middle N becomes positive

    # 9. Generate and optimize 3D structure
    ion_mol = product_mol.GetMol()
    ion_mol.UpdatePropertyCache(strict=False)
    AllChem.EmbedMolecule(ion_mol, randomSeed=0xf00d)
    AllChem.MMFFOptimizeMolecule(ion_mol)

    # Get combined indices
    idx_alkene_C1_combined = idx_alkene_C1 + n_hydrazine
    idx_alkene_C2_combined = idx_alkene_C2 + n_hydrazine

    # 10. Create TS for cycloreversion
    ts_mol = Chem.RWMol(ion_mol)
    conf = ts_mol.GetConformer()

    # 11. Break two bonds in the 5-member ring
    ts_mol.RemoveBond(idx_alkene_C1_combined, idx_alkene_C2_combined)  # Original decene double bond
    ts_mol.RemoveBond(idx_N_plus, idx_C_iminium)  # Bond between middle N and iminium C

    # 12. Change two bonds to double
    # Bond between terminal nitrogen and alkene C1
    bond_term_C1 = ts_mol.GetBondBetweenAtoms(idx_N_term, idx_alkene_C1_combined)
    if bond_term_C1:
        bond_term_C1.SetBondType(BondType.DOUBLE)

    # Bond between iminium carbon and alkene C2
    bond_im_C2 = ts_mol.GetBondBetweenAtoms(idx_C_iminium, idx_alkene_C2_combined)
    if bond_im_C2:
        bond_im_C2.SetBondType(BondType.DOUBLE)

    # 13. Transfer positive charge back
    ts_mol.GetAtomWithIdx(idx_N_term).SetFormalCharge(1)  # Terminal N becomes positive
    ts_mol.GetAtomWithIdx(idx_N_plus).SetFormalCharge(0)  # Middle N becomes neutral

    # 14. Prepare TS geometry
    # Get positions of key atoms
    pos_N_plus = np.array(conf.GetAtomPosition(idx_N_plus))
    pos_N_term = np.array(conf.GetAtomPosition(idx_N_term))
    pos_C_iminium = np.array(conf.GetAtomPosition(idx_C_iminium))
    pos_C1 = np.array(conf.GetAtomPosition(idx_alkene_C1_combined))
    pos_C2 = np.array(conf.GetAtomPosition(idx_alkene_C2_combined))

    # Elongate bonds to TS distance (2.2 Å)
    ts_distance = 2.2
    vec_N_plus_C_iminium = pos_C_iminium - pos_N_plus
    if np.linalg.norm(vec_N_plus_C_iminium) > 1e-6:
        new_vec = vec_N_plus_C_iminium * (ts_distance / np.linalg.norm(vec_N_plus_C_iminium))
        conf.SetAtomPosition(idx_C_iminium, Point3D(*(pos_N_plus + new_vec)))

    vec_C1_C2 = pos_C2 - pos_C1
    if np.linalg.norm(vec_C1_C2) > 1e-6:
        new_vec = vec_C1_C2 * (ts_distance / np.linalg.norm(vec_C1_C2))
        conf.SetAtomPosition(idx_alkene_C2_combined, Point3D(*(pos_C1 + new_vec)))

    # TODO Correct the new direction and identification of the new ene atoms

    # 15. Separate fragments
    vec1 = pos_C1 - pos_N_term
    vec2 = pos_N_plus - pos_N_term
    normal = np.cross(vec2, vec1)
    if np.linalg.norm(normal) > 1e-4:
        normal = normal / np.linalg.norm(normal)
    else:
        normal = np.array([0.0, 0.0, -1.0])

    centroid = (pos_N_plus + pos_N_term + pos_C1) / 3.0

    # Find benzene atom to determine direction
    benzene_atom_idx = None
    for nbr in ts_mol.GetAtomWithIdx(idx_C_iminium).GetNeighbors():
        if nbr.GetSymbol() == "C" and nbr.GetIsAromatic():
            benzene_atom_idx = nbr.GetIdx()
            break
    '''
    # Determine displacement direction away from benzene
    if benzene_atom_idx is not None:
        pos_benzene = np.array(conf.GetAtomPosition(benzene_atom_idx))
        benzene_vec = pos_benzene - centroid
        direction = np.sign(np.dot(normal, benzene_vec)) * normal
    else: 
    '''
    direction = normal

    displacement = direction * 2.0

    # Apply displacement to decene atoms
    ene_atoms = find_all_reachable_neighbors(ts_mol.GetAtomWithIdx(idx_alkene_C2_combined), lambda atom : atom.GetNeighbors())
    ene_atoms_idx = [atom.GetIdx() for atom in ene_atoms]
    # Exclude the transferred hydrogen

    for i in ene_atoms_idx:
        pos = np.array(conf.GetAtomPosition(i))
        new_pos = pos + displacement
        conf.SetAtomPosition(i, Point3D(*new_pos))

    # 16. Generate Gaussian input with connectivity
    ts_mol.UpdatePropertyCache(strict=False)

    # Simplify by skipping explicit benzene bond setting
    # Instead, just try to Kekulize
    try:
        Chem.SanitizeMol(ts_mol)
        Chem.Kekulize(ts_mol)
    except:
        # If Kekulization fails, proceed without it
        pass

    atom_lines = []
    for i, atom in enumerate(ts_mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        atom_lines.append(f"{atom.GetSymbol():2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")

    # Generate connectivity section
    added_bonds = set()
    connectivity_lines = []
    atom_bonds = [[] for _ in range(ts_mol.GetNumAtoms())]

    for bond in ts_mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        bond_order = int(bond.GetBondTypeAsDouble())
        bond_key = tuple(sorted([atom1_idx, atom2_idx]))
        min_idx = min(atom1_idx, atom2_idx)
        max_idx = max(atom1_idx, atom2_idx)
        if bond_key not in added_bonds:
            atom_bonds[min_idx].append(f"{max_idx + 1} {bond_order}.0")
            added_bonds.add(bond_key)

    for i in range(ts_mol.GetNumAtoms()):
        line = f"{i + 1}"
        if atom_bonds[i]:
            line += " " + " ".join(atom_bonds[i])
        connectivity_lines.append(line)

    # Route section for TS optimization
    route = (
        f"# opt=(calcfc,ts,gediis,noeigentest) freq=noraman {basis}\n"
        f"geom=connectivity int=grid=ultrafine {method} temperature=298"
    )

    input_content = f"""%mem={mem}
%nprocshared={nproc}
%chk={os.path.splitext(output_file)[0]}.chk
{route}

{smiles} + dec-5-ene cycloreversion TS (E5)

{charge} {mult}
""" + "\n".join(atom_lines) + "\n\n" + "\n".join(connectivity_lines) + "\n\n"

    with open(output_file, "w") as f:
        f.write(input_content)

    return output_file


if __name__ == "__main__":
    # Your example SMILES that previously failed
    example_smiles = "C[N+](N)=CC1=CC=CC=C1"
    try:
        output_file = generate_proton_transfer_product_e4(example_smiles)
        print(f"Successfully created: {output_file}")
    except Exception as e:
        print(f"Error: {str(e)}")
