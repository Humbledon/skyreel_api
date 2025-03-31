import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import cv2
import numpy as np
import sys

def setup_model():
    """Initialise le modèle SkyReels-V1-Hunyuan-I2V"""
    try:
        model_name = "Skywork/SkyReels-V1-Hunyuan-I2V"
        print(f"Chargement du modèle {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {str(e)}")
        sys.exit(1)

def process_image(image_path):
    """Prétraite l'image d'entrée"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"L'image {image_path} n'existe pas")
        
        image = Image.open(image_path)
        # Vérifier si l'image est valide
        image.verify()
        image = Image.open(image_path)  # Réouvrir l'image après verify()
        
        # Redimensionner l'image si nécessaire
        image = image.resize((512, 512))
        return image
    except Exception as e:
        print(f"Erreur lors du traitement de l'image : {str(e)}")
        sys.exit(1)

def generate_video(model, image, output_path, num_frames=16):
    """Génère une vidéo à partir de l'image"""
    try:
        # Convertir l'image en tenseur
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0)
        
        # Générer la vidéo
        with torch.no_grad():
            output = model.generate(
                image_tensor,
                max_length=num_frames,
                num_beams=5,
                temperature=0.7,
                top_p=0.9
            )
        
        # Convertir la sortie en vidéo
        frames = output.cpu().numpy()
        
        # Créer la vidéo avec OpenCV
        height, width = frames[0].shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 8.0, (width, height))
        
        if not out.isOpened():
            raise Exception("Impossible de créer le fichier vidéo")
        
        for frame in frames:
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        out.release()
        return True
    except Exception as e:
        print(f"Erreur lors de la génération de la vidéo : {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Génère une vidéo à partir d\'une image')
    parser.add_argument('--input_path', type=str, required=True, help='Chemin vers l\'image d\'entrée')
    parser.add_argument('--output_path', type=str, required=True, help='Chemin pour sauvegarder la vidéo')
    parser.add_argument('--num_frames', type=int, default=16, help='Nombre de frames à générer')
    
    args = parser.parse_args()
    
    # Créer les dossiers nécessaires
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Initialiser le modèle
    model = setup_model()
    
    # Traiter l'image
    print("Traitement de l'image...")
    image = process_image(args.input_path)
    
    # Générer la vidéo
    print("Génération de la vidéo...")
    if generate_video(model, image, args.output_path, args.num_frames):
        print(f"Vidéo générée avec succès : {args.output_path}")
    else:
        print("Échec de la génération de la vidéo")
        sys.exit(1)

if __name__ == "__main__":
    main() 