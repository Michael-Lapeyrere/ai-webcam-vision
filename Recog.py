import cv2
import json
import os
import numpy as np
from ultralytics import YOLO

# ----------------- MÉMOIRE -----------------
class Memory:
    def __init__(self, filepath="memory.json"):
        self.filepath = filepath
        self.concepts = {}  # label -> dict(count, embedding)
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
            self.concepts[label] = {
                "count": 1,
                "embedding": embedding.tolist()
            }
        else:
            self.concepts[label]["count"] += 1
            self.concepts[label]["embedding"] = embedding.tolist()
        self.save()

    def save(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.concepts, f, ensure_ascii=False, indent=2)
        print("💾 Mémoire sauvegardée")

    def find_similar(self, embedding, threshold=0.85):
        # Compare l'embedding avec ceux déjà appris
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
STABILITY_MOVEMENT = 5
STABILITY_FRAMES = 5

def is_stable(prev_box, curr_box):
    if not prev_box:
        return False
    dx = abs(prev_box[0] - curr_box[0])
    dy = abs(prev_box[1] - curr_box[1])
    dw = abs(prev_box[2] - curr_box[2])
    dh = abs(prev_box[3] - curr_box[3])
    return dx <= STABILITY_MOVEMENT and dy <= STABILITY_MOVEMENT and dw <= STABILITY_MOVEMENT and dh <= STABILITY_MOVEMENT

def extract_embedding(box, frame):
    x1, y1, x2, y2 = map(int, box)
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros(16*16, dtype=np.float32)
    patch_gray = cv2.resize(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (16,16))
    embedding = patch_gray.flatten().astype(np.float32)
    embedding /= np.linalg.norm(embedding) + 1e-6
    return embedding

# ----------------- MAIN -----------------
def main():
    memory = Memory()
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    prev_box = None
    stable_count = 0

    print("🧠 IA visuelle interactive — v0.9.3 YOLOv8")
    print("Touches : o=oui / n=non / q=quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, verbose=False)[0]
        if len(results.boxes) == 0:
            prev_box = None
            stable_count = 0
            cv2.imshow("IA voit ça", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # On prend le premier objet détecté
        box = results.boxes.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        if is_stable(prev_box, box):
            stable_count += 1
        else:
            stable_count = 0
        prev_box = box
        cv2.imshow("IA voit ça", frame)

        if stable_count >= STABILITY_FRAMES:
            embedding = extract_embedding(box, frame)

            # Vérification mémoire
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
                # Pas reconnu → input manuel
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