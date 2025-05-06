# delete_database_entry.py
from database import CatalystDatabase


def delete_entry(smiles=None, entry_id=None):
    db = CatalystDatabase()
    if smiles:
        db.conn.execute("DELETE FROM catalysts WHERE smiles=?", (smiles,))
    elif entry_id:
        db.conn.execute("DELETE FROM catalysts WHERE id=?", (entry_id,))
    db.conn.commit()
    print("Entry deleted successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", help="SMILES string to delete")
    parser.add_argument("--id", type=int, help="Database ID to delete")
    args = parser.parse_args()

    if args.smiles or args.id:
        delete_entry(smiles=args.smiles, entry_id=args.id)
    else:
        print("Specify --smiles or --id!")