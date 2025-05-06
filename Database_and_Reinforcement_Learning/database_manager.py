# database_manager.py
import tkinter as tk
from tkinter import ttk, messagebox
from database import CatalystDatabase


class DatabaseManager:
    def __init__(self, master):
        self.master = master
        master.title("Catalyst Database Manager")

        self.db = CatalystDatabase()
        self.create_widgets()
        self.refresh_entries()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Search/Filter area
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=5)

        ttk.Label(search_frame, text="Search SMILES:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=40)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        self.search_entry.bind("<KeyRelease>", self.filter_entries)

        # Treeview for database entries
        self.tree = ttk.Treeview(main_frame, columns=('ID', 'SMILES', 'Energies', 'Valid'), show='headings')
        self.tree.heading('ID', text='ID')
        self.tree.heading('SMILES', text='SMILES')
        self.tree.heading('Energies', text='Energies (e1-e6)')
        self.tree.heading('Valid', text='Valid')
        self.tree.column('ID', width=50, anchor='center')
        self.tree.column('SMILES', width=200)
        self.tree.column('Energies', width=300)
        self.tree.column('Valid', width=60, anchor='center')
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="Refresh", command=self.refresh_entries).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Delete Selected", command=self.delete_selected).pack(side=tk.RIGHT, padx=5)

    def refresh_entries(self):
        """Reload all entries from database"""
        self.tree.delete(*self.tree.get_children())
        cursor = self.db.conn.execute("""
            SELECT id, smiles, e1, e2, e3, e4, e5, e6, is_valid 
            FROM catalysts
        """)

        for row in cursor.fetchall():
            entry_id = row[0]
            smiles = row[1]
            energies = f"{row[2]:.2f}, {row[3]:.2f}, {row[4]:.2f}, {row[5]:.2f}, {row[6]:.2f}, {row[7]:.2f}"
            is_valid = "✓" if row[8] else "✗"

            self.tree.insert('', tk.END, values=(entry_id, smiles, energies, is_valid))

    def filter_entries(self, event=None):
        """Filter entries by SMILES substring"""
        query = self.search_var.get().lower()
        for child in self.tree.get_children():
            entry_smiles = self.tree.item(child)['values'][1].lower()
            self.tree.item(child, open=(query in entry_smiles))

    def delete_selected(self):
        """Delete selected entry from database"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "No entry selected!")
            return

        entry_id = self.tree.item(selected[0])['values'][0]
        try:
            self.db.conn.execute("DELETE FROM catalysts WHERE id=?", (entry_id,))
            self.db.conn.commit()
            messagebox.showinfo("Success", "Entry deleted successfully!")
            self.refresh_entries()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete entry:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DatabaseManager(root)
    root.geometry("800x600")
    root.mainloop()