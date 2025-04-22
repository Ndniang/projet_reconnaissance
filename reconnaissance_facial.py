import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Interface utilisateur
st.markdown("""
# 🧠 DETECTION VISAGES 
Bienvenue dans notre appli ! 👋  
Utilisation :
1. Téléchargez une image contenant des visages.
2. Réglez les paramètres `scaleFactor` et `minNeighbors` si nécessaire.
3. Choisissez la couleur du rectangle pour la détection.
4. Cliquez sur **Détecter les visages** pour lancer l'analyse.
5. Enregistrez l'image détectée si vous le souhaitez.
""")

# Téléversement de l'image
uploaded_file = st.file_uploader("📤 Téléchargez une image", type=["jpg", "jpeg", "png"])
color = st.color_picker("🎨 Choisissez la couleur du rectangle", "#00FF00")
minNeighbors = st.slider("🔍 Ajustez le paramètre minNeighbors", 1, 10, 5)
scaleFactor = st.slider("🔍 Ajustez le paramètre scaleFactor", 1.01, 1.5, 1.1, step=0.01)

# Chargement du classificateur
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialisation de la variable 'faces' à vide pour éviter les erreurs
faces = []

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    if st.button("👁️ Détecter les visages"):
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors
        )

        st.success(f"{len(faces)} visage(s) détecté(s).")

        for (x, y, w, h) in faces:
            color_bgr = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            cv2.rectangle(img_np, (x, y), (x + w, y + h), color_bgr, 2)

        st.image(img_np, caption='🖼️ Image avec visages détectés', use_container_width=True)


        result_image = Image.fromarray(img_np)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        result_image.save(temp_file.name)

        with open(temp_file.name, "rb") as file:
            st.download_button(
                label="📥 Télécharger l'image détectée",
                data=file.read(),  # Lire le contenu ici pour le détacher du fichier
                file_name="visages_detectes.png",
                mime="image/png"
            )

        # ✅ Fermer d'abord le fichier, puis supprimer
        temp_file.close()
        os.remove(temp_file.name)
