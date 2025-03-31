#!/bin/bash

# Mise à jour du système
apt-get update
apt-get upgrade -y

# Installation des dépendances système
apt-get install -y python3-pip python3-dev git

# Création de l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installation des dépendances Python
pip install --upgrade pip
pip install -r requirements.txt

# Création des dossiers nécessaires
mkdir -p input_images output_videos

# Donner les permissions d'exécution au script principal
chmod +x generate_video.py

echo "Installation terminée avec succès!"
echo "Pour utiliser le script, activez l'environnement virtuel avec: source venv/bin/activate" 