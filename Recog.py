import cv2
import json
import os
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ----------------- EMBEDDING -----------------
model_embed = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

def extract_embedding(box, frame):
    x1, y1, x2, y2 = map(int, box)
    patch = frame[y1:y2, x1:x2]

    if patch.size == 0:
        return np.zeros(1280)

    patch = cv2.resize(patch, (224,224))
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch = np.expand_dims(patch, axis=0)
    patch = preprocess_input(patch)

    embedding = model_embed.predict(patch, verbose=0)[0]
    embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
    return embedding

# ----------------- MÉMOIRE -----------------
class Memory:
    def __init__(self, filepath="memory.json"):
        self.filepath = filepath
        self.concepts = {}
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        self.concepts = json.loads(content)
                    else:
                        self.concepts = {}
            except json.JSONDecodeError:
                print("⚠️ memory.json corrompu ou vide, réinitialisation")
                self.concepts = {}
        else:
            self.concepts = {}
        print("📂 Mémoire chargée depuis le disque")

    def add(self, label, embedding):
        label = label.lower()
        if label not in self.concepts:
            self.concepts[label] = {"count": 1, "embedding": embedding.tolist()}
        else:
            self.concepts[label]["count"] += 1
            self.concepts[label]["embedding"] = embedding.tolist()
        self.save()

    def save(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.concepts, f, ensure_ascii=False, indent=2)
        print("💾 Mémoire sauvegardée")

    def find_similar(self, embedding, threshold=0.85):
        for label, info in self.concepts.items():
            mem_emb = np.array(info["embedding"])
            sim = np.dot(mem_emb, embedding)/(np.linalg.norm(mem_emb)*np.linalg.norm(embedding))
            if sim >= threshold:
                return label, sim
        return None, 0.0

    def print_concepts(self):
        if self.concepts:
            print("📊 Concepts connus :", ", ".join(f"{k}({v['count']})" for k,v in self.concepts.items()))
        else:
            print("📊 Aucun concept appris")

# ----------------- DÉTECTION STABLE -----------------
STABILITY_FRAMES = 3
MAX_DIST_CENTER = 50  # tolérance pour considérer l'objet stable

def is_stable(prev_box, curr_box):
    if prev_box is None:
        return False
    px1, py1, px2, py2 = prev_box
    cx1, cy1, cx2, cy2 = curr_box
    prev_cx, prev_cy = (px1 + px2)/2, (py1 + py2)/2
    curr_cx, curr_cy = (cx1 + cx2)/2, (cy1 + cy2)/2
    dist = ((prev_cx - curr_cx)**2 + (prev_cy - curr_cy)**2)**0.5
    return dist < MAX_DIST_CENTER

# ----------------- MAIN -----------------
def main():
    memory = Memory()
    model = YOLO("yolov8s.pt")
    cap = cv2.VideoCapture(0)

    stable_count = 0
    prev_box = None
    target_box = None

    print("🧠 IA visuelle interactive — v0.9.3 YOLOv8s")
    print("Touches : o=oui / n=non / q=quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, verbose=False)[0]

        if len(results.boxes) == 0:
            prev_box = None
            stable_count = 0
        else:
            # Choix de l'objet le plus proche du centre
            frame_h, frame_w = frame.shape[:2]
            center_x, center_y = frame_w/2, frame_h/2

            best_box = None
            best_dist = float("inf")
            for box in results.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.tolist())
                bx, by = (x1+x2)/2, (y1+y2)/2
                dist = ((bx-center_x)**2 + (by-center_y)**2)**0.5
                if dist < best_dist:
                    best_dist = dist
                    best_box = box.tolist()
                # Tous les objets en bleu clair
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 1)

            box = best_box
            x1, y1, x2, y2 = map(int, box)
            # Box suivi en vert
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, "TRACKING", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if is_stable(prev_box, box):
                stable_count += 1
            else:
                stable_count = 1
            prev_box = box

            if stable_count >= STABILITY_FRAMES:
                target_box = box

        # Affichage du LOCKING TARGET en jaune si stable
        if target_box is not None:
            tx1, ty1, tx2, ty2 = map(int, target_box)
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0,255,255), 2)
            cv2.putText(frame, "LOCKING TARGET", (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.imshow("IA voit ça", frame)

        # Interaction utilisateur
        if stable_count >= STABILITY_FRAMES and target_box is not None:
            embedding = extract_embedding(target_box, frame)
            sim_label, sim_score = memory.find_similar(embedding)
            if sim_label:
                user_input = input(f"👉 C'est '{sim_label}' ? (o/n) : ").strip().lower()
                if user_input == 'o':
                    memory.add(sim_label, embedding)
                    print(f"✅ Concept '{sim_label}' confirmé automatiquement")
                elif user_input == 'n':
                    new_label = input("👉 Nouveau label : ").strip().lower()
                    if new_label:
                        memory.add(new_label, embedding)
            else:
                user_input = input("👉 Qu'est-ce que c'est ? : ").strip().lower()
                if user_input:
                    memory.add(user_input, embedding)
            memory.print_concepts()
            stable_count = 0

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()