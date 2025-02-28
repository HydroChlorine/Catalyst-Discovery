from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import argparse
import os
import math


def visualize_smiles(input_file, output_file, rows=None, cols=5,
                     image_size=(600, 600), font_size=20, dpi=300):
    """
    Generate high-resolution grid image of chemical structures
    """
    # Read SMILES from file
    with open(input_file, 'r') as f:
        smiles_list = [line.strip() for line in f.readlines()]

    # Generate molecule objects
    mols = []
    valid_smiles = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mols.append(mol)
            valid_smiles.append(smiles)

    # Calculate grid dimensions
    if rows is None:
        rows = math.ceil(len(mols) / cols)
    else:
        cols = math.ceil(len(mols) / rows)

    # Create high-resolution image
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=cols,
        subImgSize=image_size,
        legends=valid_smiles,
        legendFontSize=font_size,
        returnPNG=False
    )

    # Save with quality settings
    img.save(output_file, dpi=(dpi, dpi), quality=100)
    print(f"Generated high-res visualization ({cols}x{rows})")
    print(f"Image dimensions: {img.size[0]}x{img.size[1]} pixels")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='High-res SMILES visualization')
    parser.add_argument('-i', '--input', required=True, help='Input SMILES file')
    parser.add_argument('-o', '--output', default='structures.png',
                        help='Output image file (png recommended)')
    parser.add_argument('--rows', type=int, help='Number of rows in grid')
    parser.add_argument('--cols', type=int, default=3,
                        help='Number of columns in grid (default: 3)')
    parser.add_argument('--size', type=int, default=600,
                        help='Image size for each structure in pixels (default: 600)')
    parser.add_argument('--font', type=int, default=20,
                        help='Legend font size (default: 20)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for output image (default: 300)')

    args = parser.parse_args()

    visualize_smiles(
        input_file=args.input,
        output_file=args.output,
        rows=args.rows,
        cols=args.cols,
        image_size=(args.size, args.size),
        font_size=args.font,
        dpi=args.dpi
    )