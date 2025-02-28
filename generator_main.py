from smiles_generator import RecursiveSMILESGenerator
from tqdm import tqdm
import time


def main():
    generator = RecursiveSMILESGenerator(
        max_depth=100,
        ring_prob=0.99,
        min_ring_size=3,
        max_ring_size=10
    )

    num_samples = 200
    timeout = 60  # seconds

    print("🚀 Starting SMILES generation with ring formation...")
    start_time = time.time()

    smiles_list = []
    with tqdm(total=num_samples, desc="Generating") as pbar:
        while len(smiles_list) < num_samples and (time.time() - start_time) < timeout:
            smiles = generator.generate_smiles()
            if smiles and smiles not in smiles_list:
                smiles_list.append(smiles)
                pbar.update(1)

    with open("generated_structures.smi", "w") as f:
        f.write("\n".join(smiles_list))

    print(f"\n🎉 Successfully generated {len(smiles_list)} structures!")
    print(f"⏱  Total time: {time.time() - start_time:.2f} seconds")
    print(f"💾 Saved to: generated_structures.smi")


if __name__ == "__main__":
    main()