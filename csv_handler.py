import pandas as pd

def process_csv(command: str, csv_file) -> dict:
    """
    Process the uploaded CSV file based on the command ("shophouse" or "industrial").

    Args:
        command (str): The command, either "shophouse" or "industrial".
        csv_file: The uploaded CSV file object.

    Returns:
        dict: A dictionary containing the processed data.
            - For "shophouse": {"addresses": [...], "primary_approved_use": [...], "secondary_approved_use": [...]}
            - For "industrial": {"addresses": [...]}
    """
    csv_file.seek(0)
    df = pd.read_csv(csv_file, header=0)  # Skip header row

    if command.lower() == "shophouse":
        addresses = df.iloc[:, 0].dropna().tolist()
        primary_approved_use = df.iloc[:, 1].dropna().tolist() if df.shape[1] > 1 else []
        secondary_approved_use = df.iloc[:, 2].dropna().tolist() if df.shape[1] > 2 else []
        return {
            "addresses": addresses,
            "primary_approved_use": primary_approved_use,
            "secondary_approved_use": secondary_approved_use
        }
    elif command.lower() == "industrial":
        addresses = df.iloc[:, 0].dropna().tolist()
        return {
            "addresses": addresses
        }
    else:
        raise ValueError("Invalid command. Must be 'shophouse' or 'industrial'.")