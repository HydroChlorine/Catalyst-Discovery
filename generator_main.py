from smiles_generator import RecursiveSMILESGenerator


def main():
    generator = RecursiveSMILESGenerator(max_depth=7)

    # Generate 10 SMILES with 60-second timeout
    smiles_list = generator.generate_multiple_smiles(
        num_samples=20,
        timeout=60
    )

    # Save results
    with open("generated_smiles.txt", "w") as f:
        f.write("\n".join(smiles_list))

    print(f"\nGenerated {len(smiles_list)} valid structures!")


if __name__ == "__main__":
    main()