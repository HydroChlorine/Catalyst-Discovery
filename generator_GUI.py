import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from generator_main import main as generate_structures
from smiles_SDF_converter import smi_to_sdf
from smiles_visualizer import visualize_smiles
import os


class CatalystGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Catalyst SMILES Generator")
        self.create_widgets()
        self.set_defaults()

    def create_widgets(self):
        # Configuration Frame
        config_frame = ttk.LabelFrame(self.root, text="Generator Parameters")
        config_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Base SMILES
        ttk.Label(config_frame, text="Base SMILES:").grid(row=0, column=0, sticky="w")
        self.base_smiles = ttk.Entry(config_frame, width=40)
        self.base_smiles.grid(row=0, column=1, columnspan=2, sticky="ew")

        # Numerical Parameters
        params = [
            ("Max Depth", "max_depth", 15),
            ("Initial Depth", "initial_depth", 2),
            ("Samples", "num_samples", 200),
            ("Depth Increment Step", "depth_step", 20),
            ("Ring Probability", "ring_prob", 0.3)
        ]

        for i, (label, var_name, default) in enumerate(params):
            ttk.Label(config_frame, text=f"{label}:").grid(row=i + 1, column=0, sticky="w")
            entry = ttk.Entry(config_frame, width=10)
            entry.grid(row=i + 1, column=1, sticky="w")
            setattr(self, var_name, entry)

        # Output Options
        output_frame = ttk.LabelFrame(self.root, text="Output Options")
        output_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(output_frame, text="Output File:").grid(row=0, column=0, sticky="w")
        self.output_entry = ttk.Entry(output_frame, width=30)
        self.output_entry.grid(row=0, column=1, sticky="ew")
        ttk.Button(output_frame, text="Browse", command=self.browse_output).grid(row=0, column=2, padx=5)

        self.generate_image = tk.BooleanVar()
        ttk.Checkbutton(output_frame, text="Generate Structure Image",
                       variable=self.generate_image).grid(row=1, column=0, columnspan=3, sticky="w")

        # Visualization options
        ttk.Label(output_frame, text="Output Format:").grid(row=2, column=0, sticky="w")
        self.output_format = ttk.Combobox(output_frame, values=["PDF", "Unsupported now"], width=12)
        self.output_format.grid(row=2, column=1, sticky="w")

        ttk.Label(output_frame, text="Max per Page:").grid(row=3, column=0, sticky="w")
        self.max_per_page = ttk.Entry(output_frame, width=5)
        self.max_per_page.grid(row=3, column=1, sticky="w")

        self.generate_sdf = tk.BooleanVar()
        ttk.Checkbutton(output_frame, text="Generate SDF File",
                       variable=self.generate_sdf).grid(row=4, column=0, columnspan=3, sticky="w")

        self.add_numbers = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_frame, text="Add Line Numbers",
                       variable=self.add_numbers).grid(row=5, column=0, columnspan=3, sticky="w")

        # Control Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=2, column=0, pady=10)
        ttk.Button(button_frame, text="Generate", command=self.run_generation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=5)

    def set_defaults(self):
        self.base_smiles.insert(0, r"[*]/[N+](N[*])=C\C1=CC=CC=C1")
        self.max_depth.insert(0, "15")
        self.initial_depth.insert(0, "2")
        self.num_samples.insert(0, "200")
        self.depth_step.insert(0, "20")
        self.ring_prob.insert(0, "0.3")
        self.output_entry.insert(0, "generated_structures.smi")
        self.output_format.set("PDF")
        self.max_per_page.insert(0, "30")

    def browse_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".smi",
            filetypes=[("SMILES files", "*.smi"), ("All files", "*.*")]
        )
        if path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, path)

    def run_generation(self):
        try:
            raw_smiles = self.base_smiles.get()
            processed_smiles = raw_smiles.replace("R", "*")

            params = {
                'base_smiles': processed_smiles,
                'max_depth': int(self.max_depth.get()),
                'initial_depth': int(self.initial_depth.get()),
                'num_samples': int(self.num_samples.get()),
                'depth_step': int(self.depth_step.get()),
                'ring_prob': float(self.ring_prob.get()),
                'output_file': self.output_entry.get()
            }

            # Run generator
            generate_structures_custom(**params)

            # Post-processing
            base_path = os.path.splitext(params['output_file'])[0]

            if self.generate_sdf.get():
                sdf_path = f"{base_path}.sdf"
                smi_to_sdf(params['output_file'], sdf_path)

            if self.generate_image.get():
                img_path = f"{base_path}.{self.output_format.get().lower()}"

                # Safe directory creation
                img_dir = os.path.dirname(img_path)
                if img_dir:  # Only create if path contains directories
                    os.makedirs(img_dir, exist_ok=True)

                visualize_smiles(
                    input_file=params['output_file'],
                    output_file=img_path,
                    output_format=self.output_format.get(),
                    add_numbers=self.add_numbers.get(),
                    max_per_page=int(self.max_per_page.get())
                )

            messagebox.showinfo("Success", "Generation completed successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")


def generate_structures_custom(base_smiles, output_file, num_samples=200, initial_depth=2,
                              max_depth=15, depth_step=20, ring_prob=0.3):
    from smiles_generator import RecursiveSMILESGenerator
    from tqdm import tqdm
    from rdkit import Chem

    # Validate base SMILES
    base_mol = Chem.MolFromSmiles(base_smiles)
    if not base_mol:
        raise ValueError(f"Invalid base SMILES: {base_smiles}")

    generator = RecursiveSMILESGenerator(
        base_smiles=base_smiles,
        max_depth=max_depth,
        ring_prob=ring_prob
    )

    results = []
    with tqdm(total=num_samples, desc="Generating") as pbar:
        while len(results) < num_samples:
            current_count = len(results)
            current_depth = min(
                initial_depth + (current_count // depth_step),
                max_depth
            )
            generator.max_depth = current_depth

            smiles = generator.generate_smiles()
            if smiles and smiles not in results:
                results.append(smiles)
                pbar.update(1)
                pbar.set_description(f"Generating (Depth {current_depth})")

    with open(output_file, "w") as f:
        f.write("\n".join(results))

    print(f"Generated {len(results)} structures")


if __name__ == "__main__":
    root = tk.Tk()
    app = CatalystGeneratorGUI(root)
    root.mainloop()