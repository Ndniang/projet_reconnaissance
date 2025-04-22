import streamlit as st
import cv2 # cv2 ( OpenCV) est une bibliothèque puissante pour le traitement d'image et la vision par ordinateur
import numpy as np
from PIL import Image  # le module PIL (Python Imaging Library), qui est une bibliothèque de traitement d’image en Python
import tempfile  #qui permet de créer des fichiers et dossiers temporaires en Python
import os  #qui sert à interagir avec le système d’exploitation (Operating System).
#Ça permet de manipuler des fichiers, dossiers, chemins, variables d’environnement

#Ajoutez des instructions à l’interface de l’application Streamlit pour guider l’utilisateur sur la façon d’utiliser l’application.

st.markdown("""
Bienvenue dans l'application de détection de visages ! 👋  
Voici comment l'utiliser :
1. Téléchargez une image contenant des visages.
2. Réglez les paramètres `scaleFactor` et `minNeighbors` si nécessaire.
3. Choisissez la couleur du rectangle pour la détection.
4. Cliquez sur **Détecter les visages** pour lancer l'analyse.
5. Enregistrez l'image détectée si vous le souhaitez.
""")

#Ajoutez une fonctionnalité pour enregistrer les images avec les visages détectés sur l'appareil de l'utilisateur
uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

#Ajoutez une fonctionnalité permettant à l'utilisateur de choisir la couleur des rectangles dessinés autour des visages détectés.
color = st.color_picker("Choisissez la couleur du rectangle", "#00FF00")

#Ajoutez une fonctionnalité pour ajuster le paramètre minNeighbors dans la fonction face_cascade.detectMultiScale().
minNeighbors = st.slider("Ajustez le paramètre minNeighbors", 1, 10, 5)

#Ajoutez une fonctionnalité pour ajuster le paramètre scaleFactor dans la fonction face_cascade.detectMultiScale().
scaleFactor = st.slider("Ajustez le paramètre scaleFactor", 1.01, 1.5, 1.1, step=0.01)
