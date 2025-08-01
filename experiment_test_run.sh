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
    sed -i "s/^$name.*/$name $value/" "$config_file"

    sed -i "s/^laser_type.*/laser_type expA/" "$config_file"
    ./build_no_mpi/hhg params/experiment.config

    sed -i "s/^laser_type.*/laser_type expB/" "$config_file"
    ./build_no_mpi/hhg params/experiment.config

    sed -i "s/^laser_type.*/laser_type exp/" "$config_file"
    ./build_no_mpi/hhg params/experiment.config
done
