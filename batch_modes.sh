#!/bin/bash
input_file="params/for_auto.txt"
readarray -t NEW_VALUES < "${input_file}"

declare -A TOKENS
TOKENS=(
  [1]="v_F"
  [v]="v_F"
  [2]="band_width"
  [W]="band_width"
  [3]="field_amplitude"
  [A]="field_amplitude"
  [4]="T"
  [T]="T"
  [5]="E_F"
  [E]="E_F"
  [6]="diagonal_relaxation_time"
  [t]="diagonal_relaxation_time"
  [7]="offdiagonal_relaxation_time"
  [o]="offdiagonal_relaxation_time"
)

echo "Select the parameter that is to be varied:"
echo "1 / v: v_F (Fermi velocity)"
echo "2 / W: band_width (W)"
echo "3 / A: field_amplitude (E_0)"
echo "4 / T: T (Temperature)"
echo "5 / E: E_F (Fermi Energy)"
echo "6 / t: diagonal_relaxation_time (tau_D)"
echo "7 / o: offdiagonal_relaxation_time (tau_C)"
read -p "Enter your choice: " choice
TOKEN=${TOKENS[$choice]}

if [ -z "$TOKEN" ]; then
  echo "Invalid choice. Exiting."
  exit 1
fi

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")

rm -rf auto_generated_${CURRENT_TIME}/
mkdir -p auto_generated_${CURRENT_TIME}

for NEW_VALUE in "${NEW_VALUES[@]}"; do
  NEW_NAME=$(echo "$NEW_VALUE" | sed 's/ /_/g')
  # Loop through each line in the config file
  while read line; do
    if [[ $line == \#* ]]; then
      continue
    fi

    # Split the line into token and value
    TOKEN_NAME=$(echo "$line" | awk '{print $1}')
    TOKEN_VALUE=$(echo "$line" | cut -d' ' -f2-)

    if [[ "$TOKEN_NAME" == "$TOKEN" ]]; then
      # replace the value with the new one
      sed "s/$TOKEN_NAME $TOKEN_VALUE/$TOKEN_NAME $NEW_VALUE/" params/cluster.config > auto_generated_${CURRENT_TIME}/$NEW_NAME.config
      break
    fi
  done < params/cluster.config
  cp slurm/hhg.slurm auto_generated_${CURRENT_TIME}/$NEW_NAME.slurm
  sed -i "s|#SBATCH --job-name=hhg|#SBATCH --job-name=${CURRENT_TIME}_$NEW_NAME|" auto_generated_${CURRENT_TIME}/$NEW_NAME.slurm
  sed -i "s|#SBATCH --output=/home/althueser/phd/cpp/HigherHarmonics/output.txt|#SBATCH --output=/home/althueser/phd/cpp/HigherHarmonics/${CURRENT_TIME}_output_$NEW_NAME.txt|" auto_generated_${CURRENT_TIME}/$NEW_NAME.slurm
  sed -i "s|mpirun ./build_cluster/hhg params/cluster.config|mpirun ./build_cluster/hhg auto_generated_${CURRENT_TIME}/$NEW_NAME.config|" auto_generated_${CURRENT_TIME}/$NEW_NAME.slurm

  # Execute the program
  sbatch auto_generated_${CURRENT_TIME}/$NEW_NAME.slurm
done
