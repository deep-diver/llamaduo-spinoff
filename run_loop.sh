#!/bin/bash

# Check for sufficient arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <path/to/python_script.py> <number_of_iterations>"
    exit 1
fi

# Get iterations and script path
python_script=$1
iterations=$2

# Validate iteration count
if ! [[ $iterations =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid input: Please provide a positive integer for the number of iterations."
    exit 1
fi

# Check if Python script exists
if [ ! -f "$python_script" ]; then
    echo "Python script not found: $python_script"
    exit 1
fi

# Check if it's evaluation.py and modify seed if so
if [[ $python_script == *"data_gen.py"* ]]; then
    yaml_file="config/synth_data_gen.yaml"  # Update with your actual YAML file path
    if [ ! -f "$yaml_file" ]; then
        echo "YAML file not found: $yaml_file"
        exit 1
    fi

    # Generate a random seed between 0 and 99999
    new_seed=$((RANDOM % 100000))

    # Modify the YAML file based on the OS
    if [[ "$(uname)" == "Darwin" ]]; then  # macOS
        sed -i '' "s/seed: .*/seed: $new_seed/" "$yaml_file"
    else  # Linux (and other Unix-like systems)
        sed -i "s/seed: .*/seed: $new_seed/" "$yaml_file"
    fi
    echo "Seed in $yaml_file changed to $new_seed for evaluation.py"
fi

# Loop to run the Python script
for (( i=1; i<=$iterations; i++ )); do
    echo "Running iteration $i of $python_script..."
    python "$python_script"
    echo "Iteration $i completed."
    echo ""
done

echo "All iterations finished!"
