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
        ttk.Label(config_frame, text="Max Allowed Difference:").grid(row=1, column=0, sticky="w")
        self.max_diff_entry = ttk.Entry(config_frame, width=8)
        self.max_diff_entry.grid(row=1, column=1, sticky="w")
        self.max_diff_entry.insert(0, "0.04")

        # ===== Candidate Frame Content =====
        self.candidate_tree = ttk.Treeview(candidate_frame,
                                           columns=("SMILES", "Status", "ΔE21", "ΔE54"),
                                           show="headings")
        self.candidate_tree.heading("SMILES", text="SMILES")
        self.candidate_tree.heading("Status", text="Status")
        self.candidate_tree.heading("ΔE21", text="E2-E1")
        self.candidate_tree.heading("ΔE54", text="E5-E4")
        self.candidate_tree.column("SMILES", width=200)
        self.candidate_tree.column("Status", width=80)
        self.candidate_tree.column("ΔE21", width=80, anchor='center')
        self.candidate_tree.column("ΔE54", width=80, anchor='center')
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
        try:
            # Get max difference from GUI
            max_diff = float(self.max_diff_entry.get())

            # Initialize generator with parameters
            self.generator = RLGenerator(
                db=self.db,
                model=self.predictor,
                max_diff=max_diff,  # Pass the parameter
                base_smiles=self.base_smiles.get(),
                max_depth=15,
                ring_prob=0.3
            )

            # Get candidates with predictions and differences
            self.current_candidates = self.generator.generate_optimized(n=5)
            self.update_candidate_list()

            if not self.current_candidates:
                messagebox.showinfo("Info",
                                    "No candidates met the criteria. Try increasing max difference.")

        except Exception as e:
            messagebox.showerror("Error", f"Generation failed:\n{str(e)}")

    def update_candidate_list(self):
        """Refresh candidate display with energy differences"""
        self.candidate_tree.delete(*self.candidate_tree.get_children())
        for smi, preds, diffs in self.current_candidates:  # Modified unpacking
            status = "New" if not self.db.exists(smi) else "Calculated"
            self.candidate_tree.insert('', tk.END, values=(
                smi,
                status,
                f"{diffs[0]:.3f}",
                f"{diffs[1]:.3f}"
            ))

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
            is_valid = True

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