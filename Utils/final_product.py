from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType
from Utils.find_all_reachable_neighbors import find_all_reachable_neighbors
import os

def generate_final_product(smiles, output_file = "final_product_e6.com", charge = 1, mult = 1,
                           mem = "180GB", nproc = 40, method = "m062x", basis = "6-311+g(2d,p)"):
    """
    Generate gaussian input file with coordinates and connectivity for the final product (new hydrazine)
    :param smiles: The input SMILES string which represent the initial hydrazine
    :param output_file: Name of output file
    :param charge: Charge on the initial molecule, assign manually because rdkit may go wrong
    :param mult: Multiplicity of the molecule
    :param mem: Memory allocation
    :param nproc: Number of processors
    :param method: Method of calculation
    :param basis: Basis set
    """

    hydrazine_mol = Chem.MolFromSmiles(smiles)
    if not hydrazine_mol:
        raise ValueError(f"Invalid SMILES: {smiles}")

    hydrazine_mol = Chem.AddHs(hydrazine_mol)
    AllChem.EmbedMolecule(hydrazine_mol, randomSeed = 0xf00d)
    AllChem.MMFFOptimizeMolecule(hydrazine_mol)

    n_plus_candidates = [atom for atom in hydrazine_mol.GetAtoms()
                         if atom.GetSymbol() == "N" and atom.GetFormalCharge() == 1]

    if not n_plus_candidates:
        raise ValueError("No positive charged nitrogen found in molecule")

    n_plus = None
    terminal_n = None
    iminium_c = None

    for candidate in n_plus_candidates:
        neighbors = candidate.GetNeighbors()
        n_nbrs = 0
        c_nbrs = 0

        for nbr in neighbors:
            bond = hydrazine_mol.GetBondBetweenAtoms(nbr.GetIdx(), candidate.GetIdx())
            if nbr.GetSymbol() == "N" and bond.GetBondType() == BondType.SINGLE:
                terminal_n = nbr
                n_nbrs += 1
            if nbr.GetSymbol() == "C" and bond.GetBondType() == BondType.DOUBLE:
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

    benzene_ring = False
    for nbr in iminium_c.GetNeighbors():
        if nbr.GetSymbol() == "C" and nbr.GetIsAromatic():
            benzene_ring = True
            break

    if not benzene_ring:
        raise ValueError("Iminium carbon must be attached to a benzene ring")

    # The above code ensures identifying the hydrazine atoms, in extremely complex molecules may produce wrong results
    idx_N_plus = n_plus.GetIdx()
    idx_N_term = terminal_n.GetIdx()
    idx_C_iminium = iminium_c.GetIdx()

    # Creating the dec-5-ene
    decene_mol = Chem.MolFromSmiles("CCCCC=CCCCC")
    decene_mol = Chem.AddHs(decene_mol)
    AllChem.EmbedMolecule(decene_mol, randomSeed = 0xf00d)
    AllChem.MMFFOptimizeMolecule(decene_mol)

    # Find double bond
    double_bond = next((bond for bond in decene_mol.GetBonds()
                        if bond.GetBondType() == BondType.DOUBLE), None)

    if not double_bond:
        raise ValueError("No double bond found in decene(?), need to check rdkit and AllChem")
    idx_alkene_C1 = double_bond.GetBeginAtomIdx()
    idx_alkene_C2 = double_bond.GetEndAtomIdx()

    # Create combined molecule
    combined = Chem.CombineMols(hydrazine_mol, decene_mol)
    product_mol = Chem.RWMol(combined)
    n_hydrazine = hydrazine_mol.GetNumAtoms()

    # Add new bonds
    product_mol.AddBond(idx_N_term, idx_alkene_C1 + n_hydrazine, BondType.SINGLE)
    product_mol.AddBond(idx_C_iminium, idx_alkene_C2 + n_hydrazine, BondType.SINGLE)

    # Modify existing bond
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

    # Adjust charges
    product_mol.GetAtomWithIdx(idx_N_plus).SetFormalCharge(0)
    product_mol.GetAtomWithIdx(idx_N_term).SetFormalCharge(1)

    # Proton transfer
    terminal_n_atom = product_mol.GetAtomWithIdx(idx_N_term)
    h_to_transfer = None
    for nbr in terminal_n_atom.GetNeighbors():
        if nbr.GetSymbol() == "H":
            h_to_transfer = nbr
            break

    if not h_to_transfer:
        raise ValueError("No hydrogen found on terminal nitrogen for proton transfer")

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
    new_h = Chem.Atom("H")
    new_h_idx   = product_mol.AddAtom(new_h)
    product_mol.AddBond(new_h_idx, idx_N_plus, BondType.SINGLE)

    # Break other bonds to form product
    idx_alkene_C1_combined = idx_alkene_C1 + n_hydrazine
    idx_alkene_C2_combined = idx_alkene_C2 + n_hydrazine

    product_mol.RemoveBond(idx_alkene_C1_combined, idx_alkene_C2_combined)
    product_mol.RemoveBond(idx_N_plus, idx_C_iminium)

    bond_term_C1 = product_mol.GetBondBetweenAtoms(idx_N_term, idx_alkene_C1_combined)
    if bond_term_C1:
        bond_term_C1.SetBondType(BondType.DOUBLE)

    bond_im_C2 = product_mol.GetBondBetweenAtoms(idx_C_iminium, idx_alkene_C2_combined)
    if bond_im_C2:
        bond_im_C2.SetBondType(BondType.DOUBLE)

    ene_atoms = find_all_reachable_neighbors(product_mol.GetAtomWithIdx(idx_alkene_C2_combined), lambda atom : atom.GetNeighbors())
    ene_atoms_idx = [atom.GetIdx() for atom in ene_atoms]

    for ene_atom in ene_atoms:
        product_mol.RemoveAtom(ene_atom.GetIdx())

    product_mol.UpdatePropertyCache(strict = False)
    AllChem.EmbedMolecule(product_mol, randomSeed=0xf00d)
    if not product_mol.GetRingInfo().AreRingFamiliesInitialized():
        Chem.AssignStereochemistry(product_mol, cleanIt = True, force = True)
        Chem.SetAromaticity(product_mol, Chem.AromaticityModel.AROMATICITY_MDL)
    AllChem.MMFFOptimizeMolecule(product_mol)
    try:
        Chem.SanitizeMol(product_mol)
        Chem.Kekulize(product_mol)
    except:
        pass

    conf = product_mol.GetConformer()
    atom_lines = []
    for i, atom in enumerate(product_mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        atom_lines.append(f"{atom.GetSymbol():2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")

    added_bonds = set()
    connectivity_lines = []
    atom_bonds = [[] for _ in range(product_mol.GetNumAtoms())]
    for bond in product_mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        bond_order = bond.GetBondTypeAsDouble()
        bond_key = tuple(sorted([atom1_idx,atom2_idx]))
        min_idx = min(atom1_idx, atom2_idx)
        max_idx = max(atom1_idx, atom2_idx)
        if bond_key not in added_bonds:
            atom_bonds[min_idx].append(f"{max_idx+1} {bond_order}")
            added_bonds.add(bond_key)

    for i in range(product_mol.GetNumAtoms()):
        line = f"{i + 1}"
        if atom_bonds[i]:
            line += " " + " ".join(atom_bonds[i])
        connectivity_lines.append(line)

    route = (
        f"# {basis} scrf=(solvent=Dichloromethane,CPCM) geom=connectivity\n"
        f"empiricaldispersion=gd3 int=grid=ultrafine {method} sp"
    )

    input_content = f"""%mem={mem}
%nprocshared={nproc}
%chk={os.path.splitext(output_file)[0]}.chk
{route}

{smiles} final product
Gibbs free energy calculation

{charge} {mult}
""" + "\n".join(atom_lines) + "\n\n" + "\n".join(connectivity_lines) + "\n\n"

    with open(output_file, "w") as f:
        f.write(input_content)

    return output_file


if __name__ == "__main__":

    example_smiles = "C[N+](N)=CC1=CC=CC=C1"
    try:
        output_file = generate_final_product(example_smiles)
        print(f"Successfully created: {output_file}")
    except Exception as e:
        print(f"Error {str(e)}")
