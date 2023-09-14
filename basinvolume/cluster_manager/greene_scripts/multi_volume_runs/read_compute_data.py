



import os
import pandas as pd

def extract_directory_info(folder_name):
    """Extract information from the folder name."""
    parts = folder_name.split('_')
    
    # Extract minimizer_name
    minimizer_name = parts[0]
    
    # Extract packing_fraction and id, if it exists.
    if len(parts) == 4:
        packing_fraction = float(parts[2])
        id_ = int(parts[3])
    else:
        packing_fraction = float(parts[2])
        id_ = 0

    return minimizer_name, packing_fraction, id_

def extract_jammed_info(jammed_folder_name):
    """Extract information from the jammed folder name."""
    return jammed_folder_name.split('explore_bv_')[1]

def get_data_from_file(path, file_name, read_function):
    """Read data from a file using a provided read function."""
    file_path = os.path.join(path, file_name)
    if os.path.exists(file_path):
        return read_function(file_path)
    return {}

def read_volume_data(filename):
    data_dict = {}
    current_section = None

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith("#") or not line:
                continue
            
            # Check for section headers
            if line.startswith("[") and line.endswith("]"):
                section_name = line[1:-1]
                data_dict[section_name] = {}
                current_section = section_name
            else:
                key, value = line.split(":")
                data_dict[current_section][key.strip()] = float(value.strip())

    return data_dict

def scan_directory(base_directory, read_function=read_volume_data):
    """Scan the provided directory and construct the DataFrame."""
    rows = []

    for folder in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder)
        if os.path.isdir(folder_path):
            minimizer_name, packing_fraction, id_ = extract_directory_info(folder)

            for jammed_folder in os.listdir(folder_path):
                if "explore_bv_" in jammed_folder:
                    jammed_packing_name = extract_jammed_info(jammed_folder)

                    analysis_path = os.path.join(folder_path, jammed_folder, "analysis")
                    data_dict = get_data_from_file(analysis_path, "mbar_volume_data", read_function)
                    print(folder)
                    print(jammed_folder)
                    print(data_dict)
                    
                    if len(data_dict) == 0:
                        continue
                    data_dict = data_dict["VOLUME_MBAR"]
                    # Construct the row
                    row = {
                        "minimizer_name": minimizer_name,
                        "jammed_packing_fname": jammed_packing_name,
                        "packing_fraction": packing_fraction,
                        "id": id_
                    }
                    row.update(data_dict)
                    rows.append(row)

    # Convert rows to DataFrame
    df = pd.DataFrame(rows)
    return df




if __name__ == "__main__":
    filename = "/scratch/ps4586/volume_runs_multi_packing_new/LBFGS_32_0.87_11/explore_bv_jammed_packing0/analysis/mbar_volume_data"
    result = read_volume_data(filename)
    folder = "/scratch/ps4586/volume_runs_multi_packing_new/"
    res = scan_directory(folder)
    
    res.to_hdf("/scratch/ps4586/volume_data.hdf5", key="volume_data")
    
    print(res)