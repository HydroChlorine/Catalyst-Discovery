import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from database import CatalystDatabase
from model_trainer import EnergyPredictor
from rl_generator import RLGenerator
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageTk
import os


class CatalystApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Catalyst Optimization System")
        self.db = CatalystDatabase()
        self.predictor = EnergyPredictor()
        self._ensure_model_trained()
        self.generator = None
        self.current_candidates = []

        # Initialize GUI components
        self.create_widgets()
        self.load_model()

        # Configure grid weights
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

    def create_widgets(self):
        """Create and arrange GUI components"""
        # Control Frame
        control_frame = ttk.LabelFrame(self.root, text="Controls")
        control_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        # Configuration Frame
        config_frame = ttk.LabelFrame(self.root, text="Configuration")
        config_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        # Candidate Frame
        candidate_frame = ttk.LabelFrame(self.root, text="Generated Candidates")
        candidate_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")

        # ===== Control Frame Content =====
        ttk.Button(control_frame, text="Generate Candidates",
                   command=self.generate_candidates).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load Model",
                   command=self.load_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save Model",
                   command=self.save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Train Model",
                   command=self._ensure_model_trained).pack(side=tk.LEFT, padx=5)

        # ===== Configuration Frame Content =====
        # Base SMILES
        ttk.Label(config_frame, text="Base SMILES:").grid(row=0, column=0, sticky="w")
        self.base_smiles = ttk.Entry(config_frame, width=40)
        self.base_smiles.grid(row=0, column=1, columnspan=2, sticky="ew")
        self.base_smiles.insert(0, r"[*]/[N+](N[*])=C\C1=CC=CC=C1")

        # Target Ranges
        ttk.Label(config_frame, text="Target Energy Ranges:").grid(row=1, column=0, sticky="w")
        self.target_ranges = []
        for i in range(6):
            ttk.Label(config_frame, text=f"E{i + 1}:").grid(row=2 + i, column=0, sticky="e")
            low_entry = ttk.Entry(config_frame, width=8)
            high_entry = ttk.Entry(config_frame, width=8)
            low_entry.grid(row=2 + i, column=1, padx=2)
            high_entry.grid(row=2 + i, column=2, padx=2)
            self.target_ranges.append((low_entry, high_entry))

        # Initialize default ranges
        default_ranges = [(-10.5, -9.5), (-5.2, -4.8), (2.1, 2.5),
                          (-3.0, -2.5), (1.8, 2.2), (-0.5, 0.5)]
        for entry, (low, high) in zip(self.target_ranges, default_ranges):
            entry[0].insert(0, str(low))
            entry[1].insert(0, str(high))

        # ===== Candidate Frame Content =====
        self.candidate_tree = ttk.Treeview(candidate_frame, columns=("SMILES", "Status"), show="headings")
        self.candidate_tree.heading("SMILES", text="SMILES")
        self.candidate_tree.heading("Status", text="Status")
        self.candidate_tree.pack(fill=tk.BOTH, expand=True)

        input_frame = ttk.Frame(candidate_frame)
        input_frame.pack(fill=tk.X, pady=5)

        ttk.Button(input_frame, text="Submit Energies",
                   command=self.submit_energies).pack(side=tk.RIGHT, padx=5)
        self.energy_entries = []
        for i in range(6):
            ttk.Label(input_frame, text=f"E{i + 1}:").pack(side=tk.LEFT, padx=2)
            entry = ttk.Entry(input_frame, width=8)
            entry.pack(side=tk.LEFT, padx=2)
            self.energy_entries.append(entry)

    def load_model(self):
        """Load existing prediction model"""
        path = filedialog.askopenfilename(filetypes=[("Joblib files", "*.joblib")])
        if path:
            try:
                self.predictor = EnergyPredictor.load(path)
                messagebox.showinfo("Success", "Model loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")

    def save_model(self):
        """Save current prediction model"""
        path = filedialog.asksaveasfilename(defaultextension=".joblib")
        if path:
            try:
                self.predictor.save(path)
                messagebox.showinfo("Success", "Model saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model:\n{str(e)}")

    # In the CatalystApp class of main_gui.py
    def _ensure_model_trained(self):
        """Train model with clearer validation"""
        X, y = self.db.get_training_data()

        if not X:
            msg = "No valid training data!\n" \
                  "1. Generate candidates\n" \
                  "2. Select a candidate\n" \
                  "3. Enter ALL 6 energy values\n" \
                  "4. Click 'Submit Energies'"
            messagebox.showwarning("Training Data Required", msg)
            return

        try:
            self.predictor.train(X, y)
            messagebox.showinfo("Success",
                                f"Model trained on {len(X)} complete entries")
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")

    def _is_model_trained(self):
        """Check if model has been trained/loaded"""
        return hasattr(self.predictor.model, "estimators_")  # MultiOutputRegressor attribute

    def generate_candidates(self):
        """Generate new candidate structures"""
        try:
            # Initialize generator with current parameters
            target_ranges = [
                (float(low.get()), float(high.get()))
                for low, high in self.target_ranges
            ]

            self.generator = RLGenerator(
                db=self.db,
                model=self.predictor,
                base_smiles=self.base_smiles.get(),
                target_ranges=target_ranges,
                max_depth = 15,
                ring_prob = 0.3
            )

            # Generate and display candidates
            self.current_candidates = self.generator.generate_optimized(n=5)
            self.update_candidate_list()

        except Exception as e:
            messagebox.showerror("Error", f"Generation failed:\n{str(e)}")

    def update_candidate_list(self):
        """Refresh candidate display"""
        self.candidate_tree.delete(*self.candidate_tree.get_children())
        for smi in self.current_candidates:
            status = "New" if not self.db.exists(smi) else "Calculated"
            self.candidate_tree.insert("", tk.END, values=(smi, status))

    def submit_energies(self):
        """Submit energy values for selected candidate"""
        selected = self.candidate_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "No candidate selected")
            return

        try:
            # Get entered energy values
            energies = [float(entry.get()) for entry in self.energy_entries]
            if len(energies) != 6:
                raise ValueError("Exactly 6 energy values required")

            # Get selected SMILES
            smi = self.candidate_tree.item(selected[0])['values'][0]

            # Validate SMILES
            if not Chem.MolFromSmiles(smi):
                raise ValueError("Invalid SMILES structure")

            # Save to database
            is_valid = all(
                low <= e <= high
                for e, (low, high) in zip(energies, [
                    (float(low.get()), float(high.get()))
                    for low, high in self.target_ranges
                ])
            )

            self.db.add_entry(smi, energies=energies, is_valid=is_valid)
            self.update_candidate_list()
            messagebox.showinfo("Success", "Energies submitted successfully")

            # Retrain model if enough data
            if len(self.db.get_training_data()[0]) % 50 == 0:
                X, y = self.db.get_training_data()
                self.predictor.train(X, y)
                messagebox.showinfo("Info", "Model updated with new data")

        except Exception as e:
            messagebox.showerror("Error", f"Submission failed:\n{str(e)}")

        try:
            self._ensure_model_trained()  # Directly call training check
        except Exception as e:
            messagebox.showwarning("Training Warning", f"Model update failed: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CatalystApp(root)
    root.mainloop()