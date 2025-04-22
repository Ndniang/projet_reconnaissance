import streamlit as st
import cv2  # OpenCV pour la détection de visages
import numpy as np
from PIL import Image  # Pour ouvrir l'image dans un format compatible avec OpenCV
import tempfile  # Pour créer un fichier temporaire si besoin
import os  # Pour gérer les fichiers sur le système

# Instructions pour l'utilisateur
st.markdown("""
Bienvenue dans l'application de détection de visages ! 👋  
Voici comment l'utiliser :
1. Téléchargez une image contenant des visages.
2. Réglez les paramètres `scaleFactor` et `minNeighbors` si nécessaire.
3. Choisissez la couleur du rectangle pour la détection.
4. Cliquez sur **Détecter les visages** pour lancer l'analyse.
5. Enregistrez l'image détectée si vous le souhaitez.
""")

# Upload de l'image
uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

# Choix de la couleur du rectangle
color = st.color_picker("Choisissez la couleur du rectangle", "#00FF00")

# Réglage du paramètre minNeighbors
minNeighbors = st.slider("Ajustez le paramètre minNeighbors", 1, 10, 5)

# Réglage du paramètre scaleFactor
scaleFactor = st.slider("Ajustez le paramètre scaleFactor", 1.01, 1.5, 1.1, step=0.01)

# ✅ Utilisation d’un conteneur d’affichage vide pour éviter les erreurs DOM (removeChild)
image_placeholder = st.empty()

# ✅ Ajouter un bouton clair pour déclencher la détection
if uploaded_file is not None:
    if st.button("Détecter les visages"):

        # Lire l'image en format RGB
        image = Image.open(uploaded_file)
        image_np = np.array(image.convert("RGB"))

        # Conversion en niveaux de gris
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Chargement du modèle de détection de visages Haar
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Détection des visages
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

        # Dessiner les rectangles autour des visages détectés
        for (x, y, w, h) in faces:
            # 🎨 Conversion hex -> RGB pour la couleur du rectangle
            color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            cv2.rectangle(image_np, (x, y), (x + w, y + h), color_rgb, 2)

        # ✅ Affichage sécurisé dans le placeholder
        image_placeholder.image(image_np, caption=f"{len(faces)} visage(s) détecté(s)", use_column_width=True)

        # ✅ Option pour enregistrer l'image
        if st.button("Enregistrer l'image avec visages détectés"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                save_path = tmp.name
                # Sauvegarde en BGR pour OpenCV
                cv2.imwrite(save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                with open(save_path, "rb") as file:
                    st.download_button(
                        label="📥 Télécharger l'image",
                        data=file,
                        file_name="visages_detectes.png",
                        mime="image/png"
                    )
                os.unlink(save_path)  # Nettoyage du fichier temporaire
