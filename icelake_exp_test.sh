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

# Loop over each value in the values array
for value in "${values[@]}"; do
    sed -i "s/^$name.*/$name $value/" "params/icelake_experiment_A.config"
    sed -i "s/^$name.*/$name $value/" "params/icelake_experiment_B.config"
    sed -i "s/^$name.*/$name $value/" "params/icelake_experiment.config"

    ./icelake_batcher.sh
done
