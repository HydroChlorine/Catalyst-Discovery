from rdkit import Chem
from rdkit.Chem import Draw
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
import os
import tempfile
from pathlib import Path


def visualize_smiles(input_file, output_file, output_format='pdf',
                     rows=None, cols=4, image_size=(1.5, 1.5),
                     font_size=12, add_numbers=True, max_per_page=50):
    """Generate PDF/PNG with numbered structures"""
    # Create parent directories if missing
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read SMILES with line numbers
    with open(input_file, 'r') as f:
        smiles_list = [(i + 1, line.strip()) for i, line in enumerate(f.readlines())]

    # Filter valid SMILES
    valid = [(num, smi) for num, smi in smiles_list if Chem.MolFromSmiles(smi)]
    if not valid:
        raise ValueError("No valid SMILES structures found")

    if output_format.lower() == 'pdf':
        _generate_pdf(valid, output_file, cols, image_size, font_size, add_numbers, max_per_page)
    else:
        _generate_png(valid, output_file, cols, image_size, font_size, add_numbers, max_per_page)


def _generate_pdf(data, filename, cols=5, image_size=(1.5, 1.5),
                  font_size=12, add_numbers=True, max_per_page=50):
    """Generate multi-page PDF with numbered structures"""
    # Add directory check
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    c = canvas.Canvas(filename, pagesize=A4)
    page_width, page_height = A4
    margin = 0.5 * inch
    img_width = image_size[0] * inch
    img_height = image_size[1] * inch

    x_positions = [margin + i * (img_width + 0.2 * inch)
                   for i in range(cols)]
    y_initial = page_height - margin - img_height

    x_idx = 0
    y_idx = y_initial
    page_num = 1

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, (line_num, smi) in enumerate(data):
            if idx % max_per_page == 0 and idx != 0:
                c.showPage()
                page_num += 1
                y_idx = y_initial
                x_idx = 0

            mol = Chem.MolFromSmiles(smi)
            if not mol:
                continue

            # Draw molecule
            temp_img = os.path.join(tmpdir, f"mol_{idx}.png")
            Draw.MolToFile(mol, temp_img, size=(300, 300))

            # Add image and text
            c.drawImage(temp_img, x_positions[x_idx], y_idx,
                        width=img_width, height=img_height)

            # Add SMILES string
            text = f"{smi}"
            c.setFont("Helvetica", font_size // 2)
            c.drawString(x_positions[x_idx], y_idx - 0.1 * inch, text)

            if add_numbers:
                text = f"{line_num}"
                c.setFont("Helvetica", font_size)
                c.drawString(x_positions[x_idx], y_idx - 0.25 * inch, text)

            # Update positions
            x_idx += 1
            if x_idx >= cols:
                x_idx = 0
                y_idx -= img_height + 0.5 * inch

                if y_idx < margin:
                    c.showPage()
                    page_num += 1
                    y_idx = y_initial

        c.save()


def _generate_png(data, filename, cols=5, image_size=(600, 600),
                  font_size=20, add_numbers=True, max_per_page=50):
    """Generate PNG grid images with directory creation and pagination"""
    from rdkit.Chem import Draw
    import math
    import os

    # Safely create directory if path contains one
    directory = os.path.dirname(filename)
    if directory:  # Only create if path has directory component
        os.makedirs(directory, exist_ok=True)

    # Split into pages
    chunks = [data[i:i + max_per_page]
              for i in range(0, len(data), max_per_page)]

    base_name = os.path.splitext(filename)[0]

    for page_idx, chunk in enumerate(chunks):
        # Extract molecules and numbers
        numbers = [str(num) for num, smi in chunk]
        mols = [Chem.MolFromSmiles(smi) for num, smi in chunk]

        # Calculate grid size
        n_mols = len(mols)
        rows = math.ceil(n_mols / cols)

        # Generate image
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=cols,
            subImgSize=image_size,
            legends=numbers if add_numbers else None,
            legendFontSize=font_size
        )

        # Save with page number if multiple pages
        if len(chunks) > 1:
            page_path = f"{base_name}_page{page_idx + 1}.png"
        else:
            page_path = filename

            # Safe directory creation for paginated files
        page_dir = os.path.dirname(page_path)
        if page_dir:
            os.makedirs(page_dir, exist_ok=True)

        img.save(page_path)