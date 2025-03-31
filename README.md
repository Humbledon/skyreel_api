# Image to Video Generator using SkyReels-V1-Hunyuan-I2V

Ce projet utilise le modèle SkyReels-V1-Hunyuan-I2V pour générer des vidéos à partir d'images.

## Prérequis

- Python 3.8 ou supérieur
- CUDA compatible GPU (recommandé)
- Git

## Installation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
cd [NOM_DU_REPO]
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Placez votre image source dans le dossier `input_images/`
2. Exécutez le script principal :
```bash
python generate_video.py --input_path "chemin/vers/votre/image.jpg" --output_path "chemin/vers/sortie.mp4"
```

## Configuration pour vast.ai

Pour déployer sur vast.ai :

1. Créez une instance avec une GPU compatible CUDA
2. Cloner ce repository sur l'instance
3. Installer les dépendances
4. Exécuter le script

## Structure du projet

```
.
├── README.md
├── requirements.txt
├── generate_video.py
├── input_images/
└── output_videos/
```

## Licence

MIT License 