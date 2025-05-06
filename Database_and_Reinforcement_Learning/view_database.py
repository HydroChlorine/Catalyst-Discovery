# view_database.py
from database import CatalystDatabase

def print_database():
    db = CatalystDatabase()
    cursor = db.conn.execute("""
        SELECT id, smiles, e1, e2, e3, e4, e5, e6
        FROM catalysts
    """)

    print("\nDatabase Entries:")
    print("ID | SMILES | Energies (e1-e6)")
    for row in cursor.fetchall():
        print(f"{row[0]} | {row[1]} | {row[2:8]}")

if __name__ == "__main__":
    print_database()