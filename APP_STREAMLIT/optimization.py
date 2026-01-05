import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import cv2
import itertools

# -------- Custom Transform: CenterPadCrop --------
class CenterPadCrop:
    """Ridimensiona mantenendo aspect ratio e aggiunge padding centrato per ottenere quadrato finale"""
    def __init__(self, final_size=256):
        self.final_size = final_size

    def __call__(self, img: Image.Image):
        # --- Resize lato lungo a final_size ---
        w, h = img.size
        if h > w:
            new_h = self.final_size
            new_w = int(w * self.final_size / h)
        else:
            new_w = self.final_size
            new_h = int(h * self.final_size / w)
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)

        # --- Calcola padding per rendere quadrato centrato ---
        pad_left = (self.final_size - new_w) // 2
        pad_right = self.final_size - new_w - pad_left
        pad_top = (self.final_size - new_h) // 2
        pad_bottom = self.final_size - new_h - pad_top

        # --- Applica padding ---
        img = transforms.functional.pad(img, padding=(pad_left, pad_top, pad_right, pad_bottom), fill=0)

        return img

def load_wheel_image(file_bytes, transform_rgb, edge_transform):
    img_array = np.frombuffer(file_bytes, np.uint8)
    img_cv2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # BGR

    # Converti BGR → RGB
    img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

    # Edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # converte a scala di grigi
    #gray_rgb = np.stack([gray]*3, axis=-1)
    edges = cv2.Canny(gray, 30, 90) # rilevamento bordi con Canny
    edges_rgb = np.stack([edges]*3, axis=-1)  # convert to 3 channels per compatibilità con edge_transform

    img = Image.fromarray(img) # converte a PIL Image
    img = ImageOps.exif_transpose(img)
    #gray_rgb = Image.fromarray(gray_rgb)
    edges_rgb = Image.fromarray(edges_rgb)
    edges_rgb = ImageOps.exif_transpose(edges_rgb)

    img = transform_rgb(img)
    edges_rgb = edge_transform(edges_rgb)

    # Concatenate RGB image and edge image along channel dimension
    combined = torch.cat((img, edges_rgb), dim=0) # concatenazione lungo la dimensione dei canali, avremo 6 canali

    return combined

# -------- Lista di 8 ruote da processare --------

positions_mapping = {
    '1s': (9, 9),
    '2s': (9, 9),
    '3s': (8, 8),
    '4s': (7, 6),
    '1d': (10, 2),
    '2d': (10, 2),
    '3d': (9, 2),
    '4d': (8, 2)
}
# --- Pre-calcolo punteggi flip/non-flip ---
# precalc_scores[wheel][pos] = (score_normale, score_flippata)
precalc_scores = {}
NUM_WHEELS = 8
position_keys = list(positions_mapping.keys())
wheel_names = []  # Questa variabile deve essere popolata con i nomi delle ruote caricate

# --- Variabili per la ricerca ---
best_score = float("inf")  # vogliamo minimizzare
best_assignment = None

# ============================================================
# Funzione ricorsiva con branch-and-bound
# ============================================================

def search(position_keys, wheel_names, precalc_scores, NUM_WHEELS, assigned_idx=0, current_perm=[], current_score=0):
    global best_score, best_assignment

    if assigned_idx == NUM_WHEELS:
        # Fine ricorsione: tutte le posizioni assegnate
        if current_score < best_score:
            best_score = current_score
            best_assignment = current_perm.copy()
        return

    pos = position_keys[assigned_idx]

    # Proviamo tutte le ruote non ancora assegnate
    for wheel in wheel_names:
        if wheel in [w for w, _ in current_perm]:
            continue  # già assegnata

        score_normale, score_flippata = precalc_scores[wheel][pos]

        # --- Prova orientamento normale ---
        total_score = current_score + score_normale
        if total_score < best_score:  # Branch and bound
            current_perm.append((wheel, 0))  # 0 = normale
            search(
                position_keys=position_keys, 
                wheel_names=wheel_names, 
                precalc_scores=precalc_scores, 
                NUM_WHEELS=NUM_WHEELS, 
                assigned_idx=assigned_idx + 1, 
                current_perm=current_perm, 
                current_score=total_score
                )
            current_perm.pop()

        # --- Prova orientamento flippato ---
        total_score = current_score + score_flippata
        if total_score < best_score:
            current_perm.append((wheel, 1))  # 1 = flippata
            search(
                position_keys=position_keys, 
                wheel_names=wheel_names, 
                precalc_scores=precalc_scores, 
                NUM_WHEELS=NUM_WHEELS, 
                assigned_idx=assigned_idx + 1, 
                current_perm=current_perm, 
                current_score=total_score
                )
            current_perm.pop()

    return best_assignment
# ============================================================
# Avvio ricerca
# ============================================================