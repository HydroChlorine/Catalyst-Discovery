import sqlite3
from rdkit import Chem
from rdkit.Chem import Descriptors


class CatalystDatabase:
    """Database manager for catalyst candidates"""

    def __init__(self, db_path="catalysts.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        """Create database table structure"""
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS catalysts (
            id INTEGER PRIMARY KEY,
            smiles TEXT UNIQUE NOT NULL,
            e1 FLOAT, e2 FLOAT, e3 FLOAT,
            e4 FLOAT, e5 FLOAT, e6 FLOAT,
            is_valid BOOLEAN,
            mol_weight FLOAT
        )""")
        self.conn.commit()

    def add_entry(self, smiles, energies=None, is_valid=None):
        """Add new entry to database"""
        mol = Chem.MolFromSmiles(smiles)
        mol_weight = Descriptors.MolWt(mol) if mol else None

        self.conn.execute("""
        INSERT OR IGNORE INTO catalysts 
        (smiles, e1, e2, e3, e4, e5, e6, is_valid, mol_weight)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (smiles, *(energies or [None] * 6), is_valid, mol_weight))
        self.conn.commit()

    def get_training_data(self):
        """Retrieve training data: features (X) and target values (y)"""
        cursor = self.conn.execute("""
        SELECT mol_weight, e1, e2, e3, e4, e5, e6
        FROM catalysts 
        WHERE e1 IS NOT NULL
          AND e2 IS NOT NULL
          AND e3 IS NOT NULL
          AND e4 IS NOT NULL
          AND e5 IS NOT NULL
          AND e6 IS NOT NULL
        """)
        data = cursor.fetchall()
        return [row[:1] for row in data], [row[1:] for row in data]

    def exists(self, smiles):
        """Check if SMILES exists in database"""
        cursor = self.conn.execute(
            "SELECT EXISTS(SELECT 1 FROM catalysts WHERE smiles=?)",
            (smiles,)
        )
        return bool(cursor.fetchone()[0])