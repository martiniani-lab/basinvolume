#!/usr/bin/env python
"""
Script to calculate and add energy and pressure to existing jammed packing config files.
Updates the original .config files with calculated energy and pressure values.
"""

import os
import argparse
import configparser
import numpy as np
import logging
from pathlib import Path
import ast
from pele.potentials import HS_WCA, InversePowerStillingerCut
from pele.utils._pressure_tensor import pressure_tensor
from pele.distance import Distance
from basinvolume.enums import Interaction
from basinvolume.utils import import_packing


def calculate_energy_pressure(config_path):
    """
    Calculate energy and pressure for a jammed packing and update the config file.
    
    Parameters
    ----------
    config_path : str or Path
        Path to the .config file
        
    Returns
    -------
    bool
        True if successfully updated, False otherwise
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logging.error(f"Config file does not exist: {config_path}")
        return False
    
    if not str(config_path).endswith('.config'):
        logging.warning(f"Skipping non-config file: {config_path}")
        return False
    
    # Determine the corresponding coordinate file
    base_name = config_path.stem  # e.g., "jammed_packing0"
    parent_dir = config_path.parent
    
    # Look for .xyzdr (3D) or .xydr (2D) file
    xyzdr_path = parent_dir / f"{base_name}.xyzdr"
    xydr_path = parent_dir / f"{base_name}.xydr"
    
    coord_path = None
    if xyzdr_path.exists():
        coord_path = xyzdr_path
        bdim = 3
    elif xydr_path.exists():
        coord_path = xydr_path
        bdim = 2
    else:
        logging.error(f"No coordinate file found for {config_path}")
        return False
    
    try:
        # Read the config file
        config = configparser.ConfigParser()
        config.read(config_path)
        
        if 'JAMMED_PACKING' not in config:
            logging.error(f"No JAMMED_PACKING section in {config_path}")
            return False
        
        # Check if energy and pressure already exist
        if config.has_option('JAMMED_PACKING', 'energy') and config.has_option('JAMMED_PACKING', 'pressure'):
            logging.info(f"Energy and pressure already exist in {config_path}, skipping")
            return False
        
        # Extract parameters from config
        nparticles = config.getint('JAMMED_PACKING', 'nparticles')
        boxdim = config.getint('JAMMED_PACKING', 'boxdim')
        ndim = nparticles * boxdim
        boxv_str = config.get('JAMMED_PACKING', 'boxv')
        boxv = np.array([float(x) for x in boxv_str.split()])
        
        distance_method = Distance[config.get('JAMMED_PACKING', 'distance_method', fallback='PERIODIC')]
        interaction_str = config.get('JAMMED_PACKING', 'interaction', fallback='HS_WCA')
        interaction = Interaction[interaction_str]
        
        pot_kwargs = ast.literal_eval(config.get('JAMMED_PACKING', 'pot_kwargs', fallback='{}'))
        sca = config.getfloat('JAMMED_PACKING', 'sca')
        
        # Read coordinates and radii from the coordinate file
        packing_data = import_packing(str(coord_path), jammed=True, bdim=boxdim)
        coords = packing_data['coords']
        hs_radii = packing_data['hs_radii']
        
        # Create the potential based on interaction type
        if interaction == Interaction.HS_WCA:
            eps = pot_kwargs.get('eps', 1.0)
            potential = HS_WCA(
                eps=eps,
                sca=sca,
                radii=hs_radii,
                boxvec=boxv,
                ndim=boxdim,
                distance_method=distance_method,
                pot_kwargs=pot_kwargs
            )
        elif interaction == Interaction.INVERSE_POWER_STILLINGER:
            stillinger_a_radii = hs_radii * (1 + sca)
            pow = pot_kwargs['pow']
            rcut = pot_kwargs['rcut']
            potential = InversePowerStillingerCut(
                pow,
                stillinger_a_radii,
                ndim=boxdim,
                boxvec=boxv,
                rcut=rcut,
                use_cell_lists=True
            )
        elif interaction == Interaction.INVERSE_POWER_HS:
            pow = pot_kwargs['pow']
            eps = pot_kwargs.get('eps', 1.0)
            sigma = pot_kwargs.get('sigma', sca)
            potential = InversePowerHS(
                pow=pow,
                eps=eps,
                sigma=sigma,
                radii=hs_radii,
                ndim=boxdim,
                boxvec=boxv,
                use_cell_lists=False
            )
        elif interaction == Interaction.INVERSE_POWER:
            # For INVERSE_POWER, we need to reconstruct the potential from the bidisperse setup
            from basinvolume.inverse_power_soft.soft_sphere_ensemble import setup_bidisperse
            parameters = pot_kwargs.copy()
            parameters['radii'] = hs_radii
            parameters['box_length'] = boxv[0]  # Assuming cubic box
            potential = setup_bidisperse(parameters, seed=parameters.get('seed', 0))['potential']
        else:
            logging.error(f"Unsupported interaction type: {interaction}")
            return False
        
        # Calculate energy
        energy = potential.getEnergy(coords)
        
        # Calculate pressure
        try:
            volume = np.prod(boxv)
            scalar_pressure, ptensor = pressure_tensor(potential, coords, volume, boxdim)
            pressure = scalar_pressure
        except Exception as e:
            logging.warning(f"Failed to calculate pressure: {e}")
            pressure = None
        
        # Update the config file
        config.set('JAMMED_PACKING', 'energy', f'{energy:.16e}')
        if pressure is not None:
            config.set('JAMMED_PACKING', 'pressure', f'{pressure:.16e}')
        
        # Write the updated config back to the same file
        with open(config_path, 'w') as f:
            f.write("#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n")
            f.write("#Generate_Jammed_Packings base class input parameters\n")
            for section in config.sections():
                f.write(f"[{section}]\n")
                for option in config.options(section):
                    value = config.get(section, option)
                    f.write(f"{option}: {value}\n")
                f.write("\n")
        
        if pressure is not None:
            logging.info(f"Updated {config_path} with energy={energy:.6e}, pressure={pressure:.6e}")
        else:
            logging.info(f"Updated {config_path} with energy={energy:.6e}, pressure=N/A")
        return True
        
    except Exception as e:
        logging.error(f"Error processing {config_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_directory(directory, pattern="jammed_packing*.config"):
    """
    Process all jammed packing config files in a directory.
    
    Parameters
    ----------
    directory : str or Path
        Directory containing jammed packing files
    pattern : str
        Glob pattern for finding config files
        
    Returns
    -------
    tuple
        (number of files processed, number of successful updates)
    """
    directory = Path(directory)
    
    if not directory.exists():
        logging.error(f"Directory does not exist: {directory}")
        return 0, 0
    
    # Find all config files matching the pattern
    config_files = sorted(directory.glob(pattern))
    
    if not config_files:
        logging.warning(f"No files matching pattern '{pattern}' in {directory}")
        return 0, 0
    
    logging.info(f"Found {len(config_files)} config files in {directory}")
    
    processed = 0
    successful = 0
    
    for config_file in config_files:
        processed += 1
        if calculate_energy_pressure(config_file):
            successful += 1
    
    return processed, successful


def main():
    parser = argparse.ArgumentParser(
        description="Calculate and add energy and pressure to existing jammed packing config files"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a directory containing jammed packing files or a single config file"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="jammed_packing*.config",
        help="Glob pattern for finding config files (default: 'jammed_packing*.config')"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories for config files"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    path = Path(args.path)
    
    if path.is_file():
        # Process single file
        success = calculate_energy_pressure(path)
        if success:
            logging.info("Successfully updated config file with energy and pressure")
        else:
            logging.error("Failed to update config file")
    elif path.is_dir():
        if args.recursive:
            # Process all subdirectories recursively
            total_processed = 0
            total_successful = 0
            
            for subdir in path.rglob("*/"):
                if "jammed_packing" in subdir.name or subdir == path:
                    processed, successful = process_directory(subdir, args.pattern)
                    total_processed += processed
                    total_successful += successful
            
            logging.info(f"\nTotal: Processed {total_processed} files, updated {total_successful} files")
        else:
            # Process single directory
            processed, successful = process_directory(path, args.pattern)
            logging.info(f"\nProcessed {processed} files, updated {successful} files")
    else:
        logging.error(f"Path does not exist: {path}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())