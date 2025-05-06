# add_dummy_data.py
from database import CatalystDatabase

# Initialize database
db = CatalystDatabase()

# Add a valid catalyst entry with all 6 energies
db.add_entry(
    smiles="C",  # Simple methane (replace with your base SMILES if needed)
    energies=[-10.0, -5.0, 2.3, -2.7, 2.0, 0.0],  # Example energies
    is_valid=True
)

print("Dummy data added successfully!")