from rdkit import Chem

def smi_to_sdf(input_file, output_file):
    writer = Chem.SDWriter(output_file)
    with open(input_file) as f:
        for line in f:
            smiles = line.strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                writer.write(mol)
    print(f"Generated {output_file} with valid structures")

if __name__ == "__main__":
    import sys
    smi_to_sdf(sys.argv[1], sys.argv[2])