from smiles_generator import RecursiveSMILESGenerator

def main():
    num_samples = 20  # You can change this number
    output_file = "generated_smiles.txt"

    generator = RecursiveSMILESGenerator(max_depth=7)
    smiles_list = generator.generate_multiple_smiles(num_samples)

    # Save to a text file
    with open(output_file, "w") as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")

    print(f"Generated {num_samples} SMILES structures and saved them to '{output_file}'.")

if __name__ == "__main__":
    main()