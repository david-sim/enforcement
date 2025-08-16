"""
CSV Processing Module for Smart Compliance Operations Engine
Handles CSV file processing, validation, and data extraction with modular functions.
"""
import pandas as pd
import io
from typing import Dict, List, Tuple, Optional, Union


class CSVValidationError(Exception):
    """Custom exception for CSV validation errors"""
    pass


def detect_csv_delimiter(csv_file) -> str:
    """
    Detect the delimiter used in a CSV file (comma or semicolon).
    
    Args:
        csv_file: The uploaded CSV file object
        
    Returns:
        The detected delimiter (',' or ';')
        
    Raises:
        CSVValidationError: If delimiter detection fails
    """
    try:
        csv_file.seek(0)
        first_line = csv_file.readline()
        
        # If it's bytes, decode it
        if isinstance(first_line, bytes):
            first_line = first_line.decode('utf-8')
        
        first_line = first_line.strip()
        csv_file.seek(0)
        
        # Check if the first line contains semicolons (might be semicolon-separated)
        if first_line and ';' in first_line and ',' not in first_line:
            return ';'
        else:
            return ','
            
    except Exception as e:
        csv_file.seek(0)
        raise CSVValidationError(f"Failed to detect CSV delimiter: {str(e)}")


def load_csv_file(csv_file, delimiter: str = None) -> pd.DataFrame:
    """
    Load CSV file into a pandas DataFrame with proper encoding handling.
    
    Args:
        csv_file: The uploaded CSV file object
        delimiter: Optional delimiter override
        
    Returns:
        pandas DataFrame containing the CSV data
        
    Raises:
        CSVValidationError: If CSV loading fails
    """
    try:
        csv_file.seek(0)
        
        # Auto-detect delimiter if not provided
        if delimiter is None:
            delimiter = detect_csv_delimiter(csv_file)
        
        # Try to read with detected delimiter
        try:
            df = pd.read_csv(csv_file, sep=delimiter, header=0)
        except Exception as e:
            # Try alternative encoding
            csv_file.seek(0)
            df = pd.read_csv(csv_file, sep=delimiter, header=0, encoding='utf-8')
            
        return df
        
    except Exception as e:
        raise CSVValidationError(f"Failed to load CSV file: {str(e)}")


def validate_csv_structure(df: pd.DataFrame, address_type: str) -> Dict[str, any]:
    """
    Validate CSV structure and content based on address type.
    
    Args:
        df: pandas DataFrame to validate
        address_type: Type of processing ("shophouse" or "industrial")
        
    Returns:
        Dictionary with validation results and warnings
        
    Raises:
        CSVValidationError: If validation fails
    """
    validation_result = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    
    # Check if DataFrame is empty
    if df.empty:
        raise CSVValidationError("CSV file is empty or contains no data rows")
    
    # Check minimum columns (must have at least 1 for address)
    if len(df.columns) < 1:
        raise CSVValidationError("CSV must contain at least one column (address)")
    
    # Check maximum columns (should only have exactly 3 columns for shophouse, flexible for industrial)
    if address_type.lower() == "shophouse" and len(df.columns) > 3:
        raise CSVValidationError(f"CSV contains {len(df.columns)} columns. For shophouse processing, expected exactly 3 columns: Address, Primary Approved Use, Secondary Approved Use")
    elif address_type.lower() == "industrial" and len(df.columns) > 3:
        validation_result['warnings'].append(f"CSV contains {len(df.columns)} columns. For industrial processing, only the first column (Address) is required. Additional columns will be processed if available.")
    
    # Validate first column (addresses) - cannot be entirely empty
    if df.iloc[:, 0].isna().all():
        raise CSVValidationError("First column (addresses) cannot be entirely empty")
    
    # Count empty addresses
    empty_addresses = df.iloc[:, 0].isna().sum()
    if empty_addresses > 0:
        validation_result['warnings'].append(f"Found {empty_addresses} empty addresses that will be skipped")
    
    # Validate address formats
    addresses = df.iloc[:, 0].dropna().tolist()
    invalid_addresses = []
    addresses_with_commas = []
    
    for i, addr in enumerate(addresses):
        addr_str = str(addr).strip()
        if not addr_str:
            continue
        
        row_num = i + 2  # +2 because pandas is 0-indexed and we have a header row
            
        # Check for Singapore address format
        if not _is_valid_singapore_address(addr_str):
            invalid_addresses.append(f"Row {row_num}: '{addr_str[:50]}...' - Invalid Singapore address format")
        
        # Check for suspicious patterns that might indicate wrong data
        if any(char in addr_str for char in ['@', 'http', 'www']):
            invalid_addresses.append(f"Row {row_num}: '{addr_str[:50]}...' - Contains suspicious patterns (email/website)")
        
        # Check for commas in address data (potential CSV parsing issues)
        if ',' in addr_str:
            addresses_with_commas.append(f"Row {row_num}: '{addr_str[:50]}...' - Contains commas (may cause CSV parsing issues)")
    
    # Add invalid address warnings (limit to first 10 for readability)
    if invalid_addresses:
        validation_result['warnings'].extend(invalid_addresses[:10])
        if len(invalid_addresses) > 10:
            validation_result['warnings'].append(f"... and {len(invalid_addresses) - 10} more address format issues")
    
    # Add comma warnings
    if addresses_with_commas:
        validation_result['warnings'].extend(addresses_with_commas[:5])
        if len(addresses_with_commas) > 5:
            validation_result['warnings'].append(f"... and {len(addresses_with_commas) - 5} more addresses with comma issues")
    
    # Validate primary approved use column (column 2) - requirement depends on address type
    if address_type.lower() == "shophouse":
        # For shophouse addresses, primary approved use is required
        if len(df.columns) < 2:
            raise CSVValidationError("CSV must contain at least 2 columns for shophouse processing: Address and Primary Approved Use")
        
        primary_use_col = df.iloc[:, 1]
        
        # Check if primary approved use column is entirely empty
        if primary_use_col.isna().all():
            raise CSVValidationError("Primary approved use column (column 2) cannot be entirely empty for shophouse processing")
        
        # Count empty primary approved use entries
        primary_use_empty = primary_use_col.isna().sum()
        if primary_use_empty > 0:
            validation_result['warnings'].append(f"Found {primary_use_empty} empty primary approved use entries")
        
        # Validate that primary approved use entries are strings (not numbers or other types)
        non_string_primary_uses = []
        for i, use in enumerate(primary_use_col.dropna()):
            if pd.isna(use):
                continue
            use_str = str(use).strip()
            if not use_str:  # Empty string after stripping
                continue
            if use_str.replace('.', '').replace('-', '').isdigit():  # Looks like a number
                non_string_primary_uses.append(f"Row {i+2}: '{use_str}' - Primary approved use appears to be numeric")
        
        if non_string_primary_uses:
            validation_result['warnings'].extend(non_string_primary_uses[:5])
            if len(non_string_primary_uses) > 5:
                validation_result['warnings'].append(f"... and {len(non_string_primary_uses) - 5} more numeric primary approved use issues")
    
    elif address_type.lower() == "industrial":
        # For industrial addresses, only address column is required
        if len(df.columns) >= 2:
            # If primary approved use column exists, validate it but don't require it
            primary_use_col = df.iloc[:, 1]
            primary_use_empty = primary_use_col.isna().sum()
            
            if primary_use_empty == len(df):
                validation_result['warnings'].append("Primary approved use column (column 2) is entirely empty (this is acceptable for industrial processing)")
            elif primary_use_empty > 0:
                validation_result['warnings'].append(f"Found {primary_use_empty} empty primary approved use entries (acceptable for industrial processing)")
            
            # Validate that primary approved use entries are strings when present
            non_string_primary_uses = []
            for i, use in enumerate(primary_use_col.dropna()):
                if pd.isna(use):
                    continue
                use_str = str(use).strip()
                if not use_str:  # Empty string after stripping
                    continue
                if use_str.replace('.', '').replace('-', '').isdigit():  # Looks like a number
                    non_string_primary_uses.append(f"Row {i+2}: '{use_str}' - Primary approved use appears to be numeric")
            
            if non_string_primary_uses:
                validation_result['warnings'].extend(non_string_primary_uses[:5])
                if len(non_string_primary_uses) > 5:
                    validation_result['warnings'].append(f"... and {len(non_string_primary_uses) - 5} more numeric primary approved use issues")
    
    # Validate secondary approved use column (column 3) - optional but should be strings if present
    if len(df.columns) >= 3:
        secondary_use_col = df.iloc[:, 2]
        secondary_use_empty = secondary_use_col.isna().sum()
        
        # It's OK if secondary use is entirely empty (optional field)
        if secondary_use_empty == len(df):
            validation_result['warnings'].append("Secondary approved use column (column 3) is entirely empty (this is optional)")
        elif secondary_use_empty > 0:
            validation_result['warnings'].append(f"Found {secondary_use_empty} empty secondary approved use entries")
        
        # Validate that secondary approved use entries are strings when present
        non_string_secondary_uses = []
        for i, use in enumerate(secondary_use_col.dropna()):
            if pd.isna(use):
                continue
            use_str = str(use).strip()
            if not use_str:  # Empty string after stripping
                continue
            if use_str.replace('.', '').replace('-', '').isdigit():  # Looks like a number
                non_string_secondary_uses.append(f"Row {i+2}: '{use_str}' - Secondary approved use appears to be numeric")
        
        if non_string_secondary_uses:
            validation_result['warnings'].extend(non_string_secondary_uses[:5])
            if len(non_string_secondary_uses) > 5:
                validation_result['warnings'].append(f"... and {len(non_string_secondary_uses) - 5} more numeric secondary approved use issues")
    
    return validation_result


def _is_valid_singapore_address(address: str) -> bool:
    """
    Check if an address follows Singapore address format patterns.
    Uses pattern matching similar to generate_variant function.
    
    Args:
        address: Address string to validate
        
    Returns:
        True if address appears to be a valid Singapore address
    """
    import re
    
    address_stripped = address.strip()
    
    # Should not be too short (minimum reasonable length)
    if len(address_stripped) < 15:
        return False
    
    # Pattern 1: Unit format with Singapore - e.g., "123 Street Name #02-01 Singapore 123456"
    unit_with_singapore_pattern = re.match(r"^(\d+[A-Z]?)\s+(.+)\s+#(\d{2})-\d{2}\s+(Singapore\s+)?\d{6}$", address_stripped, re.IGNORECASE)
    if unit_with_singapore_pattern:
        return True
    
    # Pattern 2: Unit format without Singapore - e.g., "114 Lavender Street #02-62 338729"
    unit_without_singapore_pattern = re.match(r"^(\d+[A-Z]?)\s+(.+)\s+#(\d{2})-\d{2}\s+\d{6}$", address_stripped)
    if unit_without_singapore_pattern:
        return True
    
    # Pattern 3: Suffix format with Singapore - e.g., "123A Street Name Singapore 123456"
    suffix_with_singapore_pattern = re.match(r"^(\d+)([A-Z])\s+(.+)\s+(Singapore\s+)?\d{6}$", address_stripped, re.IGNORECASE)
    if suffix_with_singapore_pattern:
        return True
    
    # Pattern 4: Standard format with Singapore - e.g., "123 Street Name Singapore 123456"
    standard_with_singapore_pattern = re.match(r"^(\d+)\s+(.+)\s+(Singapore\s+)?\d{6}$", address_stripped, re.IGNORECASE)
    if standard_with_singapore_pattern:
        # Additional check: street name should contain typical Singapore street indicators
        street_name = standard_with_singapore_pattern.group(2).lower()
        singapore_street_indicators = [
            'street', 'road', 'avenue', 'lane', 'drive', 'crescent', 'place', 'park', 'way',
            'st ', ' rd ', 'ave', 'ln ', 'dr ', 'cres', 'pl ', 'pk ', 'wy ',
            'boulevard', 'blvd', 'close', 'walk', 'rise', 'view', 'terrace', 'link',
            'quay', 'wharf', 'pier', 'bridge', 'gate', 'circle', 'loop', 'turn',
            'industrial', 'business', 'commercial', 'building', 'tower', 'block', 'centre',
            'shophouse', 'shop', 'mall', 'plaza', 'square', 'court', 'garden', 'hill',
            'north', 'south', 'east', 'west', 'upper', 'lower', 'central', 'new', 'old',
            'marina', 'beach', 'bay', 'river', 'lake', 'mount', 'bukit', 'tanjong'
        ]
        
        if any(indicator in street_name for indicator in singapore_street_indicators):
            return True
    
    # Pattern 5: Just 6-digit postal code at the end (fallback pattern)
    postal_code_pattern = re.search(r'\b\d{6}\b$', address_stripped)
    if postal_code_pattern:
        # Additional checks for typical Singapore address components
        address_lower = address_stripped.lower()
        
        # Must have some form of street/location identifier
        if any(indicator in address_lower for indicator in [
            'street', 'road', 'avenue', 'lane', 'drive', 'crescent', 'place', 'park',
            'boulevard', 'close', 'walk', 'rise', 'view', 'terrace', 'link', 'way',
            'quay', 'wharf', 'pier', 'bridge', 'gate', 'circle', 'loop', 'turn',
            'industrial', 'business', 'commercial', 'building', 'tower', 'block',
            'shophouse', 'shop', 'mall', 'plaza', 'marina', 'beach', 'bay'
        ]):
            # Should not contain suspicious patterns
            if not any(char in address_stripped for char in ['@', 'http', 'www']):
                return True
    
    return False


def extract_csv_data(df: pd.DataFrame, address_type: str) -> Dict[str, List[str]]:
    """
    Extract and clean data from validated CSV DataFrame.
    
    Args:
        df: pandas DataFrame containing CSV data
        address_type: Type of processing ("shophouse" or "industrial")
        
    Returns:
        Dictionary containing extracted addresses and approved uses
    """
    # Extract addresses (remove empty ones)
    addresses = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    addresses = [addr for addr in addresses if addr]  # Remove empty strings
    
    # Extract primary approved use
    if len(df.columns) >= 2:
        primary_approved_use = df.iloc[:, 1].fillna("").astype(str).str.strip().tolist()
    else:
        primary_approved_use = [""] * len(addresses)
    
    # Extract secondary approved use  
    if len(df.columns) >= 3:
        secondary_approved_use = df.iloc[:, 2].fillna("").astype(str).str.strip().tolist()
    else:
        secondary_approved_use = [""] * len(addresses)
    
    # Ensure all lists have the same length as addresses
    primary_approved_use = primary_approved_use[:len(addresses)]
    secondary_approved_use = secondary_approved_use[:len(addresses)]
    
    # Pad with empty strings if needed
    while len(primary_approved_use) < len(addresses):
        primary_approved_use.append("")
    while len(secondary_approved_use) < len(addresses):
        secondary_approved_use.append("")
    
    return {
        "addresses": addresses,
        "primary_approved_use": primary_approved_use,
        "secondary_approved_use": secondary_approved_use
    }


def process_csv_with_validation(address_type: str, csv_file) -> Dict[str, List[str]]:
    """
    Main function to process CSV with full validation and error handling.
    
    Args:
        address_type: The command, either "shophouse" or "industrial"
        csv_file: The uploaded CSV file object
        
    Returns:
        Dictionary containing the processed data
        
    Raises:
        CSVValidationError: If validation fails
        ValueError: If address_type is invalid
    """
    # Validate address type
    if address_type.lower() not in ["shophouse", "industrial"]:
        raise ValueError("Invalid address type. Must be 'shophouse' or 'industrial'.")
    
    print(f"🔍 Processing {address_type} CSV file...")
    
    try:
        # Step 1: Load CSV file
        delimiter = detect_csv_delimiter(csv_file)
        print(f"🔍 Detected delimiter: '{delimiter}'")
        
        df = load_csv_file(csv_file, delimiter)
        print(f"📊 CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Step 2: Validate CSV structure and content
        validation_result = validate_csv_structure(df, address_type)
        
        # Print warnings if any
        if validation_result['warnings']:
            print("⚠️ Validation warnings:")
            for warning in validation_result['warnings']:
                print(f"   • {warning}")
        
        # Step 3: Extract data
        extracted_data = extract_csv_data(df, address_type)
        
        print(f"📍 Extracted {len(extracted_data['addresses'])} valid addresses for {address_type} processing")
        
        # Step 4: Final validation - ensure we have addresses to process
        if not extracted_data['addresses']:
            raise CSVValidationError("No valid addresses found in CSV file after processing")
        
        return extracted_data
        
    except CSVValidationError:
        # Re-raise validation errors
        raise
    except Exception as e:
        # Convert unexpected errors to validation errors
        raise CSVValidationError(f"Unexpected error processing CSV: {str(e)}")


def create_csv_for_download(results_data: List[List[str]]) -> io.BytesIO:
    """
    Create a CSV file buffer for download with the specified columns.
    
    Args:
        results_data: List of result rows with all required columns
    
    Returns:
        BytesIO buffer containing the CSV data
    """
    columns = [
        'address',
        'confirmed_occupant', 
        'verification_analysis',
        'primary_approved_use',
        'secondary_approved_use',
        'compliance_level',
        'rationale',
        'google_address_search_results',
        'google_address_search_results_variant',
        'confirmed_occupant_google_search_results'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(results_data, columns=columns)
    
    # Create CSV buffer
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_buffer.seek(0)
    
    return csv_buffer
