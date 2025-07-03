from rdkit import Chem
from Utils.Starting_Material_and_First_TS import smiles_to_gaussian_com_e1, generate_cycloaddition_ts_com_e2
from Utils.H_transfer_intermediate import generate_cycloaddition_product_e3
from Utils.cycloreversion_ts import generate_proton_transfer_product_e4, generate_cycloreversion_ts_e5
from Utils.final_product import generate_final_product_e6


def get_user_input():
    """Get user input for SMILES and optional parameters"""
    print("Please enter the SMILES string (required):")
    smiles = input("SMILES: ").strip()
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES: {smiles}. This SMILES can't be recognized as a molecule")

    params = {
        'charge': 1,
        'mult': 1,
        'mem': "180GB",
        'nproc': 40,
        'method': "m062x",
        'basis': "6-311+g(2d,p)"
    }

    print("\nDefault parameters:")
    for param, value in params.items():
        print(f"{param}: {value}")

    print("\nWould you like to change any parameters? (y/n)")
    if input().lower() == 'y':
        print("\nEnter parameter name and new value (e.g., 'charge 2')")
        print("Press Enter twice when done")

        while True:
            entry = input().strip()
            if not entry:
                break
            try:
                param, value = entry.split(maxsplit=1)
                if param in params:
                    # Convert numeric parameters to appropriate types
                    if param in ['charge', 'mult', 'nproc']:
                        params[param] = int(value)
                    else:
                        params[param] = value
                else:
                    print(f"Warning: Unknown parameter '{param}'")
            except ValueError:
                print("Invalid input format. Use 'parameter value'")
    params['output_file'] = f"{smiles}"
    return smiles, params


if __name__ == "__main__":
    smiles, params = get_user_input()

    params['output_file'] = f"{smiles}-ME.com"
    smiles_to_gaussian_com_e1(smiles, **params)

    params['output_file'] = f"{smiles}-CATS.com"
    generate_cycloaddition_ts_com_e2(smiles, **params)

    params['output_file'] = f"{smiles}-CAME.com"
    generate_cycloaddition_product_e3(smiles, **params)

    params['output_file'] = f"{smiles}-PTME.com"
    generate_proton_transfer_product_e4(smiles, **params)

    params['output_file'] = f"{smiles}-CRTS.com"
    generate_cycloreversion_ts_e5(smiles, **params)

    params['output_file'] = f"{smiles}-CRME.com"
    generate_final_product_e6(smiles, **params)

