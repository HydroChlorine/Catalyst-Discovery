from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType
import os

def generate_cycloaddition_product_e3(smiles, output_file="intermediate_e3.com", charge=1, mult=1,
                                 mem="180GB", nproc=40, method="m062x", basis="6-311+g(2d,p)"):
    """
    Generate Gaussian input for cycloaddition product of hydrazine derivative + dec-5-ene

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
    # 1. Load and validate hydrazine derivative
    hydrazine_mol = Chem.MolFromSmiles(smiles)
    if not hydrazine_mol:
        raise ValueError(f"Invalid SMILES: {smiles}")

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

    # 4. Create dec-5-ene molecule
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

    # 5. Create product molecule (DO NOT remove hydrogen)
    combined = Chem.CombineMols(hydrazine_mol, decene_mol)
    product_mol = Chem.RWMol(combined)
    n_hydrazine = hydrazine_mol.GetNumAtoms()

    # Add new bonds for cycloaddition
    product_mol.AddBond(terminal_n.GetIdx(), idx_alkene_C1 + n_hydrazine, BondType.SINGLE)
    product_mol.AddBond(iminium_c.GetIdx(), idx_alkene_C2 + n_hydrazine, BondType.SINGLE)

    # Modify existing bonds to match product state:
    # 1. Change hydrazine double bond (N+=C) to single
    bond_to_modify = product_mol.GetBondBetweenAtoms(n_plus.GetIdx(), iminium_c.GetIdx())
    if bond_to_modify and bond_to_modify.GetBondType() == BondType.DOUBLE:
        product_mol.RemoveBond(n_plus.GetIdx(), iminium_c.GetIdx())
        product_mol.AddBond(n_plus.GetIdx(), iminium_c.GetIdx(), BondType.SINGLE)

    # 2. Change alkene double bond to single
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

    # 6. Adjust formal charges for charge transfer
    # Original N+ becomes neutral (now has 3 single bonds)
    n_plus_atom = product_mol.GetAtomWithIdx(n_plus.GetIdx())
    n_plus_atom.SetFormalCharge(0)

    # Terminal nitrogen becomes positively charged (now has 3 bonds: to N+, to H, and to alkene)
    terminal_n_atom = product_mol.GetAtomWithIdx(terminal_n.GetIdx())
    terminal_n_atom.SetFormalCharge(1)

    # 7. Generate and optimize 3D structure
    product_mol = product_mol.GetMol()
    product_mol.UpdatePropertyCache(strict=False)  # Update valence information

    # First embed without constraints
    AllChem.EmbedMolecule(product_mol, AllChem.ETKDGv3())

    # Now add constraints and optimize with UFF
    try:
        ff = AllChem.UFFGetMoleculeForceField(product_mol)
        # Add distance constraints for the new bonds
        ff.AddDistanceConstraint(
            terminal_n.GetIdx(),
            idx_alkene_C1 + n_hydrazine,
            1.55,  # Target distance
            100.0  # Force constant
        )
        ff.AddDistanceConstraint(
            iminium_c.GetIdx(),
            idx_alkene_C2 + n_hydrazine,
            1.55,  # Target distance
            100.0  # Force constant
        )
        ff.Minimize()
    except:
        # Fallback to unconstrained optimization
        AllChem.MMFFOptimizeMolecule(product_mol)

    # Final optimization without constraints
    AllChem.MMFFOptimizeMolecule(product_mol)

    # 8. Generate Gaussian input content with connectivity section
    conf = product_mol.GetConformer()
    atom_lines = []
    for i, atom in enumerate(product_mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        atom_lines.append(f"{atom.GetSymbol():2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")

    # Generate connectivity section
    added_bonds = set()
    connectivity_lines = []
    atom_bonds = [[] for _ in range(product_mol.GetNumAtoms())]

    for bond in product_mol.GetBonds():
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

    for i in range(product_mol.GetNumAtoms()):
        line = f"{i + 1}"
        if atom_bonds[i]:
            line += " " + " ".join(atom_bonds[i])
        connectivity_lines.append(line)

    # Route section for product optimization
    route = (
        f"# {basis} scrf=(solvent=Dichloromethane,CPCM) geom=connectivity\n"
        f"empiricaldispersion=gd3 int=grid=ultrafine {method} sp"
    )

    input_content = f"""%mem={mem}
%nprocshared={nproc}
%chk={os.path.splitext(output_file)[0]}.chk
{route}

{smiles} + dec-5-ene cycloaddition product
Gibbs free energy calculation

{charge} {mult}
""" + "\n".join(atom_lines) + "\n\n" + "\n".join(connectivity_lines) + "\n\n"

    # 9. Write to file
    with open(output_file, "w") as f:
        f.write(input_content)

    return output_file


if __name__ == "__main__":
    # Your example SMILES that previously failed
    example_smiles = "C[N+](N)=CC1=CC=CC=C1"
    try:
        output_file = generate_cycloaddition_product_e3(example_smiles)
        print(f"Successfully created: {output_file}")
    except Exception as e:
        print(f"Error: {str(e)}")