#!/usr/bin/env python
"""
Script to create backup files of jammed packing configurations with energy and pressure data.
Creates .config.new files from existing .config files, preserving all data.
"""

import os
import argparse
import configparser
import glob
import logging
from pathlib import Path


def backup_config_file(config_path, force_overwrite=False):
    """
    Create a backup of a jammed packing config file with .new extension.
    
    Parameters
    ----------
    config_path : str or Path
        Path to the original .config file
    force_overwrite : bool
        Whether to overwrite existing .new files
        
    Returns
    -------
    bool
        True if backup was created successfully, False otherwise
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logging.error(f"Config file does not exist: {config_path}")
        return False
    
    if not str(config_path).endswith('.config'):
        logging.warning(f"Skipping non-config file: {config_path}")
        return False
    
    # Create backup filename with .new extension
    backup_path = Path(str(config_path) + '.new')
    
    # Check if backup already exists
    if backup_path.exists() and not force_overwrite:
        logging.info(f"Backup already exists, skipping: {backup_path}")
        return False
    
    try:
        # Read the original config file
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Check if JAMMED_PACKING section exists
        if 'JAMMED_PACKING' not in config:
            logging.warning(f"No JAMMED_PACKING section in {config_path}")
            return False
        
        # Check for energy and pressure in the config
        has_energy = config.has_option('JAMMED_PACKING', 'energy')
        has_pressure = config.has_option('JAMMED_PACKING', 'pressure')
        
        if has_energy or has_pressure:
            info_parts = []
            if has_energy:
                energy = config.get('JAMMED_PACKING', 'energy')
                info_parts.append(f"energy={energy}")
            if has_pressure:
                pressure = config.get('JAMMED_PACKING', 'pressure')
                info_parts.append(f"pressure={pressure}")
            logging.info(f"Found {', '.join(info_parts)} in {config_path}")
        else:
            logging.info(f"No energy/pressure data found in {config_path}")
        
        # Write the backup file
        with open(backup_path, 'w') as f:
            # Write header
            f.write("#BACKUP FILE - Generated from {}\n".format(config_path.name))
            f.write("#This file contains energy and pressure data from jammed packing\n")
            
            # Write all sections
            for section in config.sections():
                f.write(f"\n[{section}]\n")
                for option in config.options(section):
                    value = config.get(section, option)
                    f.write(f"{option}: {value}\n")
        
        logging.info(f"Created backup: {backup_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error processing {config_path}: {e}")
        return False


def process_directory(directory, pattern="jammed_packing*.config", force_overwrite=False):
    """
    Process all jammed packing config files in a directory.
    
    Parameters
    ----------
    directory : str or Path
        Directory containing jammed packing files
    pattern : str
        Glob pattern for finding config files
    force_overwrite : bool
        Whether to overwrite existing .new files
        
    Returns
    -------
    tuple
        (number of files processed, number of successful backups)
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
        if backup_config_file(config_file, force_overwrite):
            successful += 1
    
    return processed, successful


def main():
    parser = argparse.ArgumentParser(
        description="Create backup files (.config.new) of jammed packing configurations with energy and pressure data"
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
        "--force",
        action="store_true",
        help="Force overwrite existing .new files"
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
        success = backup_config_file(path, args.force)
        if success:
            logging.info("Successfully created backup")
        else:
            logging.error("Failed to create backup")
    elif path.is_dir():
        if args.recursive:
            # Process all subdirectories recursively
            total_processed = 0
            total_successful = 0
            
            # Find all directories containing jammed_packings
            for subdir in path.rglob("*/"):
                if "jammed_packing" in subdir.name or subdir == path:
                    processed, successful = process_directory(subdir, args.pattern, args.force)
                    total_processed += processed
                    total_successful += successful
            
            logging.info(f"\nTotal: Processed {total_processed} files, created {total_successful} backups")
        else:
            # Process single directory
            processed, successful = process_directory(path, args.pattern, args.force)
            logging.info(f"\nProcessed {processed} files, created {successful} backups")
    else:
        logging.error(f"Path does not exist: {path}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())