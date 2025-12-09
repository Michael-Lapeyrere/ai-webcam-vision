import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import load_model
import h5py

print("Librairies chargées")

# Charger le modèle entraîné
model = load_model(r"C:\Users\micha\Desktop\Pro\IA\Recognition\model.h5")
class_names = ["bouche", "Oeil"]  # S'assurer que les classes sont bien dans cet ordre

print("Model récupérer et classe définies")

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

print("Ouverture webcam")

# Stocker les dernières prédictions pour plus de stabilité
recent_preds = deque(maxlen=5)

print("Stockage des dernières prédictions")

# Charger les classificateurs Haarcascades d'OpenCV pour détecter les visages, les yeux et la bouche
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(r"C:\Users\micha\Desktop\Pro\IA\Recognition\haarcascade_mcs_mouth.xml")

print("Clasificateurs chargés")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir en niveaux de gris
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Détection des yeux
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            eye_region = roi_color[ey:ey + eh, ex:ex + ew]
            eye_resized = cv2.resize(eye_region, (96, 96)) / 255.0  # Normalisation
            eye_pred = model.predict(np.expand_dims(eye_resized, axis=0))
            eye_label = class_names[np.argmax(eye_pred)]

            # Ajouter la prédiction au buffer
            recent_preds.append(eye_label)

            # Encadrer les yeux en vert
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.putText(roi_color, eye_label, (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Détection de la bouche
        mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=10, minSize=(30, 30))
        for (mx, my, mw, mh) in mouths:
            mouth_region = roi_color[my:my + mh, mx:mx + mw]
            
            # Ajuste ici pour s'assurer que l'IA détecte la bouche (et non un autre objet)
            if mouth_region.shape[0] > 0 and mouth_region.shape[1] > 0:
                mouth_resized = cv2.resize(mouth_region, (96, 96)) / 255.0  # Normalisation
                mouth_pred = model.predict(np.expand_dims(mouth_resized, axis=0))
                mouth_label = class_names[np.argmax(mouth_pred)]

                # Ajouter la prédiction au buffer
                recent_preds.append(mouth_label)

                # Encadrer la bouche en rouge
                cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
                cv2.putText(roi_color, mouth_label, (mx, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Afficher la prédiction la plus fréquente dans les dernières détections
    if len(recent_preds) > 0:
        most_common_pred = max(set(recent_preds), key=recent_preds.count)
        cv2.putText(frame, f"Objet detecte: {most_common_pred}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Live Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

eye_pred = model.predict(np.expand_dims(eye_resized, axis=0))
print(f"Prédiction yeux : {eye_pred}")

mouth_pred = model.predict(np.expand_dims(mouth_resized, axis=0))
print(f"Prédiction bouche : {mouth_pred}")


