#!/usr/bin/env python3
"""
Comprehensive MPI F08 -> MPI F90 compatibility fix for FDS-FireX
This script:
1. Wraps all remaining USE MPI_F08 statements with #ifdef WITHOUT_MPIF08 conditionals
2. Wraps TYPE(MPI_*) declarations with conditionals to use INTEGER for MPI F90
"""

import os
import re
from pathlib import Path

SOURCE_DIR = Path(r"C:\Users\ji\Documents\fireX\fds-FireX\Source")

def fix_use_mpi_f08(content):
    """Fix USE MPI_F08 statements that are not already wrapped in conditionals."""
    lines = content.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check if this is a USE MPI_F08 line not inside a conditional
        if re.match(r'^\s*USE\s+MPI_F08\s*$', stripped, re.IGNORECASE):
            # Check if the previous line is #else (meaning it's already in a conditional)
            if i > 0 and lines[i-1].strip() == '#else':
                # Already in conditional, skip
                result.append(line)
                i += 1
                continue

            # Check if the line before that is #ifdef WITHOUT_MPIF08 with USE MPI
            if i >= 2 and '#ifdef WITHOUT_MPIF08' in lines[i-2]:
                # Already in conditional, skip
                result.append(line)
                i += 1
                continue

            # Get the indentation
            indent_match = re.match(r'^(\s*)', line)
            indent = indent_match.group(1) if indent_match else ''

            # Wrap with conditional
            result.append(f'{indent}#ifdef WITHOUT_MPIF08')
            result.append(f'{indent}USE MPI')
            result.append(f'{indent}#else')
            result.append(line)  # Original USE MPI_F08
            result.append(f'{indent}#endif')
            i += 1
        else:
            result.append(line)
            i += 1

    return '\n'.join(result)


def fix_mpi_types(content):
    """Fix TYPE(MPI_*) declarations to use INTEGER for MPI F90 interface."""
    lines = content.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check for TYPE(MPI_COMM), TYPE(MPI_GROUP), TYPE(MPI_REQUEST) declarations
        mpi_type_match = re.match(r'^(\s*)(TYPE\(MPI_(COMM|GROUP|REQUEST)\))\s*(.*)$', stripped, re.IGNORECASE)

        if mpi_type_match:
            # Check if already in a conditional
            if i > 0 and lines[i-1].strip() == '#else':
                result.append(line)
                i += 1
                continue

            if i >= 2 and '#ifdef WITHOUT_MPIF08' in lines[i-2]:
                result.append(line)
                i += 1
                continue

            # Get the indentation from original line
            indent_match = re.match(r'^(\s*)', line)
            indent = indent_match.group(1) if indent_match else ''

            # Extract type and rest of declaration
            full_match = re.match(r'^(\s*)(TYPE\(MPI_(COMM|GROUP|REQUEST)\))(\s*.*)$', line, re.IGNORECASE)
            if full_match:
                original_indent = full_match.group(1)
                mpi_type = full_match.group(2)
                rest = full_match.group(4)

                # Wrap with conditional
                result.append(f'{original_indent}#ifdef WITHOUT_MPIF08')
                result.append(f'{original_indent}INTEGER{rest}')
                result.append(f'{original_indent}#else')
                result.append(line)  # Original TYPE(MPI_*)
                result.append(f'{original_indent}#endif')
                i += 1
            else:
                result.append(line)
                i += 1
        else:
            result.append(line)
            i += 1

    return '\n'.join(result)


def clean_nested_conditionals(content):
    """Remove incorrectly nested #ifdef WITHOUT_MPIF08 blocks."""
    # Pattern to find nested conditionals that need cleanup
    # This looks for #ifdef followed soon by another #ifdef
    lines = content.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this starts a nested conditional block that should be removed
        if '#ifdef WITHOUT_MPIF08' in line:
            # Look ahead to see if we have a proper block
            if i + 4 < len(lines):
                block = [lines[j].strip() for j in range(i, min(i + 6, len(lines)))]

                # Check for badly nested blocks like:
                # #ifdef WITHOUT_MPIF08
                # #ifdef WITHOUT_MPIF08
                # USE MPI
                # ...
                if len(block) > 1 and '#ifdef WITHOUT_MPIF08' in block[1]:
                    # Skip the outer redundant #ifdef
                    i += 1
                    continue

        result.append(line)
        i += 1

    return '\n'.join(result)


def process_file(filepath):
    """Process a single Fortran file."""
    print(f"Processing: {filepath.name}")

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            original_content = f.read()
    except Exception as e:
        print(f"  Error reading: {e}")
        return False

    # Apply fixes
    content = original_content
    content = fix_use_mpi_f08(content)
    content = fix_mpi_types(content)
    content = clean_nested_conditionals(content)

    if content != original_content:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Modified!")
            return True
        except Exception as e:
            print(f"  Error writing: {e}")
            return False
    else:
        print(f"  No changes needed")
        return False


def main():
    """Main function."""
    print("=" * 60)
    print("Comprehensive MPI F08 -> MPI F90 Compatibility Fix")
    print("=" * 60)

    # Get all Fortran files
    fortran_files = list(SOURCE_DIR.glob("*.f90"))
    print(f"\nFound {len(fortran_files)} Fortran files")

    modified_count = 0
    for filepath in fortran_files:
        if process_file(filepath):
            modified_count += 1

    print(f"\n{'=' * 60}")
    print(f"Modified {modified_count} files")
    print("=" * 60)


if __name__ == "__main__":
    main()
