# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
# Main script for hyperparameter optimisation

# Funzione per convertire secondi in formato ore:minuti:secondi
format_time() {
    local seconds=$1
    printf "%02d:%02d:%02d" $((seconds/3600)) $((seconds%3600/60)) $((seconds%60))
}


# Modifiable experiment options.
# Expt options include {volatility}
EXPT=hvg
OUTPUT_FOLDER='./hvg'  # Path to store data & experiment outputs
OUTPUT_FOLDERvenv='./volatility20outputs'  # Path to store data & experiment outputs
USE_GPU=no
TESTING_MODE=no  # If yes, trains a small model with little data to test script


#export CUDA_VISIBLE_DEVICES=1  #  <--  Set GPU here!  Adjust the GPU ID as needed.

# Step 1: Setup environment.
echo
echo Setting up virtual environment...
echo

set -e


pip3.8 install virtualenv # Assumes pip3 is installed!
python3.8 -m virtualenv $OUTPUT_FOLDERvenv/venv11
source $OUTPUT_FOLDERvenv/venv11/bin/activate

pip3 install -r requirements.txt


# Step 2: Downloads data if not present.
#echo
#python3.8 -m script_download_data $EXPT $OUTPUT_FOLDER

# Nome del file di log
LOG_FILE="training_log.txt"
TIMESTAMPS_FILE="training_timestamps.txt"

# Data e ora di inizio
START_TIME=$(date +%s)
START_DATETIME=$(date "+%Y-%m-%d %H:%M:%S")

# Registra l'inizio dell'addestramento
echo "=== TRAINING LOG ===" > "$LOG_FILE"
echo "Inizio: $START_DATETIME" >> "$LOG_FILE"
echo "$START_TIME" > "$TIMESTAMPS_FILE"

# Esegue il comando di addestramento
echo "Comando eseguito: $@" >> "$LOG_FILE"
echo "-------------------" >> "$LOG_FILE"

# Step 3: Train & Test
echo
python3.8 -m script_train_fixed_params $EXPT $OUTPUT_FOLDER $USE_GPU $TESTING_MODE

# Uncomment below for full hyperparamter optimisation.
#python3.8 -m script_hyperparam_opt $EXPT $OUTPUT_FOLDER $USE_GPU yes




# Esegue il comando passato come argomento e cattura l'output
"$@" 2>&1 | while IFS= read -r line
do
    # Timestamp corrente
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    FORMATTED_TIME=$(format_time $ELAPSED)
    
    # Registra il timestamp e l'output
    echo "[$FORMATTED_TIME] $line" | tee -a "$LOG_FILE"
    echo "$CURRENT_TIME" >> "$TIMESTAMPS_FILE"
done

# Calcola e registra il tempo totale
END_TIME=$(date +%s)
END_DATETIME=$(date "+%Y-%m-%d %H:%M:%S")
TOTAL_TIME=$((END_TIME - START_TIME))
FORMATTED_TOTAL=$(format_time $TOTAL_TIME)

echo "-------------------" >> "$LOG_FILE"
echo "Fine: $END_DATETIME" >> "$LOG_FILE"
echo "Tempo totale: $FORMATTED_TOTAL" >> "$LOG_FILE"

# Stampa il riepilogo
echo ""
echo "Training completato!"
echo "Tempo totale: $FORMATTED_TOTAL"
echo "Log salvato in: $LOG_FILE"