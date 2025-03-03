from smiles_generator import RecursiveSMILESGenerator
from tqdm import tqdm
import time


def main():
    num_samples = 200
    output_file = "progressive_structures.smi"

    # Progressive depth parameters
    initial_depth = 2
    max_final_depth = 15
    depth_increment_step = 20  # Increase depth every N samples

    # Initialize generator with base depth
    generator = RecursiveSMILESGenerator(max_depth=initial_depth)

    results = []
    with tqdm(total=num_samples, desc="Generating") as pbar:
        while len(results) < num_samples:
            # Dynamically adjust max_depth based on progress
            current_count = len(results)
            current_depth = min(
                initial_depth + (current_count // depth_increment_step),
                max_final_depth
            )
            generator.max_depth = current_depth

            # Generate and validate
            smiles = generator.generate_smiles()
            if smiles and smiles not in results:
                results.append(smiles)
                pbar.update(1)

                # Update progress bar description
                pbar.set_description(f"Generating (Depth {current_depth})")

    # Save results
    with open("generated_structures.smi", "w") as f:
        f.write("\n".join(results))

    print(f"Successfully generated {len(results)} structures")
    print(f"Final complexity depth: {generator.max_depth}")


if __name__ == "__main__":
    main()