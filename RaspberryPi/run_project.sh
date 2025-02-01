#!/bin/bash
# Usage:
#   ./run_project.sh <architecture> <image_path> <mode>
#
#   <architecture> : MLP ou CNN
#   <image_path>   : chemin vers l'image à tester (par exemple images/exemple.bmp)
#   <mode>         : "train" pour entraîner le modèle et compiler, 
#                    "notrain" pour utiliser les poids déjà exportés
#
# Exemple d'utilisation :
#   ./run_project.sh CNN images/exemple.bmp train
#   ./run_project.sh MLP images/exemple.bmp notrain

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <architecture (MLP|CNN)> <image_path> <mode (train|notrain)>"
    exit 1
fi

ARCH=$1
IMAGE_PATH=$2
MODE=$3

# Définir les variables en fonction de l'architecture choisie
if [ "$ARCH" = "MLP" ]; then
    TRAIN_SCRIPT="train_MLP.py"
    WEIGHTS_FILE="mlp_weights.h"
    COMPILE_FLAG="-D USE_MLP"
elif [ "$ARCH" = "CNN" ]; then
    TRAIN_SCRIPT="train_CNN.py"
    WEIGHTS_FILE="cnn_weights.h"
    COMPILE_FLAG="-D USE_CNN"
else
    echo "Architecture inconnue : $ARCH. Utilisez MLP ou CNN."
    exit 1
fi

# Vérification du mode
if [ "$MODE" = "train" ]; then
    echo "=== Entraînement du modèle $ARCH ==="
    python3 "$TRAIN_SCRIPT"
    if [ $? -ne 0 ]; then
        echo "Erreur lors de l'entraînement du modèle $ARCH."
        exit 1
    fi
    echo "Entraînement terminé et export des poids réalisé dans $WEIGHTS_FILE."
elif [ "$MODE" = "notrain" ]; then
    echo "=== Mode non entraînement : Vérification de l'existence de $WEIGHTS_FILE ==="
    if [ ! -f "$WEIGHTS_FILE" ]; then
        echo "Erreur : Le fichier $WEIGHTS_FILE contenant les poids de l'architecture $ARCH n'existe pas."
        exit 1
    fi
    echo "Le fichier $WEIGHTS_FILE existe."
else
    echo "Mode inconnu : $MODE. Utilisez 'train' ou 'notrain'."
    exit 1
fi

# Compilation du code C d'inférence
echo "=== Compilation du code C d'inférence avec l'architecture $ARCH ==="
gcc digits_id.c -o digits_id $COMPILE_FLAG
if [ $? -ne 0 ]; then
    echo "Erreur lors de la compilation du code C."
    exit 1
fi
echo "Compilation terminée."

# Exécution du programme d'inférence sur l'image spécifiée
echo "=== Exécution de l'inférence sur l'image $IMAGE_PATH ==="
./digits_id "$IMAGE_PATH"
