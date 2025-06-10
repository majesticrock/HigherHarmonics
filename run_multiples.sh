#!/bin/bash

# Ensure that the script is executed with the correct number of arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 name value1 [value2 ...]"
    exit 1
fi

# Assign the first argument to the name variable and shift the remaining arguments to the values array
name=$1
shift
values=("$@")

# Path to the params.config file
config_file="params/experiment.config"

# Ensure that the params.config file exists
if [ ! -f "$config_file" ]; then
    echo "Error: Configuration file '$config_file' not found!"
    exit 1
fi

# Loop over each value in the values array
for value in "${values[@]}"; do
    # Replace the value corresponding to the key == name in the params.config file
    # Use sed to find the line containing the key and replace its value
    sed -i "s/^$name.*/$name $value/" "$config_file"

    # Execute the script ./exec.sh
    ./build_no_mpi/hhg params/experiment.config
done
