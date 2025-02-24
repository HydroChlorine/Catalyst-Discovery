import torch
import torch.nn as nn
import subprocess
import pandas as pd
from rdkit import Chem

# ======= Step 1: AI Model (RNN-Based SMILES Generator) =======
class SMILESRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SMILESRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

# Load AI model (replace with your trained model if available)
hidden_size = 128
input_size = 64
output_size = input_size
ai_model = SMILESRNN(input_size, hidden_size, output_size)

def generate_smiles(model, seed, max_length=100):
    generated = seed
    hidden = torch.zeros(1, 1, hidden_size)
    for _ in range(max_length):
        output, hidden = model(torch.tensor([generated]), hidden)
        next_char = torch.argmax(output).item()
        generated.append(next_char)
        if next_char == 0:  # End token
            break
    return "".join([chr(c) for c in generated])

# ======= Step 2: Running Gaussian DFT Calculations =======
def run_dft(smiles):
    """Runs Gaussian DFT calculation and extracts activation energy."""
    gaussian_input = f"""%chk=catalyst.chk
%nprocshared=8
%mem=16GB
# B3LYP/6-31G(d) Opt Freq

Generated Catalyst

0 1
N 0.0 0.0 0.0
N 0.0 0.0 1.1
C 0.0 0.0 2.2

"""

    with open("catalyst.gjf", "w") as f:
        f.write(gaussian_input)

    subprocess.run("g16 catalyst.gjf", shell=True)

    # Extract activation energy from Gaussian log file
    activation_energy = None
    with open("catalyst.log", "r") as f:
        for line in f:
            if "Activation Energy" in line:  # Replace with the actual Gaussian output keyword
                activation_energy = float(line.split()[-1])  # Example extraction

    return activation_energy

# ======= Step 3: Store DFT Results for Future Use =======
def save_dft_result(smiles, activation_energy):
    df = pd.DataFrame(columns=["SMILES", "Activation_Energy"])
    try:
        df = pd.read_csv("catalyst_data.csv")
    except FileNotFoundError:
        pass  # If the file doesn't exist, start a new one

    df = df.append({"SMILES": smiles, "Activation_Energy": activation_energy}, ignore_index=True)
    df.to_csv("catalyst_data.csv", index=False)

# ======= Step 4: Automate the Pipeline =======
def main():
    for i in range(10):  # Generate 10 candidate catalysts
        smiles = generate_smiles(ai_model, [ord(c) for c in "NN="])  # Seed with hydrazine core
        print(f"Generated Catalyst {i+1}: {smiles}")

        print("Running DFT validation with Gaussian...")
        activation_energy = run_dft(smiles)
        if activation_energy is not None:
            print(f"DFT Activation Energy: {activation_energy:.2f} kcal/mol")
            save_dft_result(smiles, activation_energy)
        else:
            print("DFT calculation failed.")

if __name__ == "__main__":
    main()