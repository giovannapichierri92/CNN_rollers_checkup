# Streamlit_App.py
# Frontend per il progetto di analisi usura delle rotelle dei pattini

import streamlit as st
import numpy as np
from logic import order_wheels
from chat_agent import ask_llm
import matplotlib.pyplot as plt
import base64
import matplotlib.patches as mpatches

# ----------------------------------------------------------
# CONFIGURAZIONE PAGINA
# ----------------------------------------------------------
st.set_page_config(
    page_title="Analisi Usura Pattini",
    page_icon="🛼",
    layout="wide"
)

# ----------------------------------------------------------
# SIDEBAR — LOGO E INFO
# ----------------------------------------------------------
with st.sidebar:
    st.image(
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcToOVSa9UX5UOv9TjVM_e76paS7mL8xgot2LA&s",
        use_container_width=True
    )
    st.markdown("**Rollers Check-Up**")
    st.write("Controlla l'usura delle tue ruote e trova l'assetto ottimale per le gare.")
    st.markdown("---")
    st.info("**Progetto AI** sviluppato da:\n\nAndrea De Tomasi\n\nGiovanna Pichierri\n\nGianluca Chiarello")
    st.markdown("---")
    st.write("Versione: **1.0 (Streamlit Frontend)**")

# ----------------------------------------------------------
# HEADER PAGINA
# ----------------------------------------------------------
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Russo+One&display=swap" rel="stylesheet">
    <h1 style='font-family: "Russo One", sans-serif;
               font-size: 110px;
               color: #15B8A6;
               text-align: top-center;
               text-shadow: 2px 2px 5px #444;'>ROLLERS CHECK-UP</h1>
""", unsafe_allow_html=True)
st.markdown("**Carica le foto delle tue ruote per valutarne l'usura e scoprire la migliore disposizione!**")
st.markdown("---")

# ----------------------------------------------------------
# UPLOAD IMMAGINI
# ----------------------------------------------------------
uploaded_files = st.file_uploader(
    "Caricamento immagini:",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png"]
)

if uploaded_files:
    # 🔧 salviamo una copia dei bytes (per evitare buffer consumati)
    uploaded_data = []
    for f in uploaded_files:
        f.seek(0)
        uploaded_data.append((f.name, f.getbuffer().tobytes(), f.type))

    st.success(f"Hai caricato {len(uploaded_data)} rotelle!")

    col1, _ = st.columns([1, 0.0001])  # solo una colonna visibile (col2 vuota per struttura)

    with col1:
        # ----------------------------------------------------------
        # 📤 Anteprima immagini prima di tutto
        # ----------------------------------------------------------
        st.subheader("📤 Anteprima Ruote Caricate")

        preview_size = 200
        html_code = "<div style='display:flex; flex-wrap:nowrap; overflow-x:auto; gap:1rem; padding:1rem; background-color:#fafafa; border:1px solid #ddd; border-radius:10px;'>"

        for name, file_bytes, file_type in uploaded_data:
            base64_img = base64.b64encode(file_bytes).decode('utf-8')
            html_code += (
                f"<div style='text-align:center; flex:0 0 auto;'>"
                f"<img src='data:image/{file_type.split('/')[1]};base64,{base64_img}' "
                f"style='width:{preview_size}px; height:auto; border-radius:8px; border:1px solid #ccc; object-fit:contain;' />"
                f"<p style='font-size:0.8rem; margin-top:0.3rem;'>{name}</p>"
                f"</div>"
            )
        html_code += "</div>"
        st.markdown(html_code, unsafe_allow_html=True)

        # ----------------------------------------------------------
        # 🔧 Analisi Ruote
        # ----------------------------------------------------------
        st.subheader("⚙️ Analisi Ruote")

        if st.button("Avvia analisi"):
            if len(uploaded_data) != 8:
                st.error("⚠️ Devi caricare esattamente 8 immagini per eseguire l’ottimizzazione.")
            else:
                st.info("Avvio calcolo migliore disposizione delle ruote... ⏳")

                import torch
                from torchvision import transforms
                from optimization import search, CenterPadCrop, positions_mapping, position_keys, load_wheel_image
                import timm
                import torch.nn as nn

                # --- MODELLO ---                
                class MobileNetV3Regressor(nn.Module):
                    def __init__(self, base_model):
                        super().__init__()
                        self.base = base_model

                    def forward(self, x):
                        x = self.base(x)
                        x = torch.sigmoid(x)
                        return x

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")              
                base_model = timm.create_model('mobilenetv3_large_100', pretrained=True)
                orig_conv = base_model.conv_stem
                new_conv = nn.Conv2d(
                    in_channels=6,
                    out_channels=orig_conv.out_channels,
                    kernel_size=orig_conv.kernel_size,
                    stride=orig_conv.stride,
                    padding=orig_conv.padding,
                    bias=orig_conv.bias is not None
                )
                with torch.no_grad():
                    new_conv.weight[:, :3] = orig_conv.weight
                    new_conv.weight[:, 3:] = orig_conv.weight

                base_model.conv_stem = new_conv
                in_features = base_model.classifier.in_features
                base_model.classifier = nn.Linear(in_features, 2)

                model = MobileNetV3Regressor(base_model).to(device)
                checkpoint = torch.load("fold_5_best_overall.pth", map_location=device)
                model.load_state_dict(checkpoint)
                model.eval()

                # --- TRASFORMAZIONI ---
                transform_rgb = transforms.Compose([
                    CenterPadCrop(final_size=224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])

                edge_transform = transforms.Compose([
                    CenterPadCrop(final_size=224),
                    transforms.ToTensor(),
                ])

                # --- PREVISIONI MODELLO ---
                predictions = {}
                for name, file_bytes, file_type in uploaded_data:
                    wheel_tensor = load_wheel_image(file_bytes, transform_rgb, edge_transform).unsqueeze(0).to(device)
                    with torch.no_grad():
                        prediction = model(wheel_tensor).cpu().numpy()
                    predictions[name] = prediction.flatten().tolist()

                # --- CALCOLI OTTIMIZZAZIONE ---
                wheel_names = list(predictions.keys())
                NUM_WHEELS = len(wheel_names)
                precalc_scores = {}
                for wheel in wheel_names:
                    wheel_left, wheel_right = predictions[wheel]
                    precalc_scores[wheel] = {}
                    for pos in position_keys:
                        wL, wR = positions_mapping[pos]
                        precalc_scores[wheel][pos] = (
                            wheel_left * wL + wheel_right * wR,
                            wheel_right * wL + wheel_left * wR
                        )

                best_assignment = search(
                    assigned_idx=0,
                    current_perm=[],
                    current_score=0,
                    position_keys=position_keys,
                    wheel_names=wheel_names,
                    precalc_scores=precalc_scores,
                    NUM_WHEELS=NUM_WHEELS
                )

                st.success("✅ Analisi completata!")
               
                # ----------------------------------------------------------
                # 🔒 Salvataggio dati in session_state per il Chatbot
                # ----------------------------------------------------------
                st.session_state.predictions = predictions
                st.session_state.best_assignment = best_assignment
                st.session_state.wheel_number_map = {w: f"Ruota {i+1}" for i, w in enumerate(predictions.keys())}
                st.session_state.position_keys = position_keys

                # Serve anche per recuperare le chiavi in fase di chat
                st.session_state.position_keys = position_keys

                # --- VISUALIZZAZIONE VALORI USURA ---
                st.markdown("### 🔎 Valori di usura rilevati per ogni ruota \n (0 = Ottima ✅, 1 = Pessima 🚨):")
                for i, (dx, sx) in enumerate(predictions.values()):
                    st.write(f"**Ruota {i+1}:** → Lato dx: {dx:.3f} | Lato sx: {sx:.3f}")
                     # aggiungo il riferimento all'immagine
                    if uploaded_files and i < len(uploaded_files):
                        st.markdown(f"<p style='color:gray; font-size:0.85rem; margin-top:-0.6rem;'>🖼️ <i>{uploaded_files[i].name}</i></p>",
                                    unsafe_allow_html=True
                        )
                    st.progress(int(np.mean([dx, sx]) * 100))

                # --- DISPOSIZIONE OTTIMALE ---
                st.markdown("---")
                st.subheader("⚡ Migliore disposizione trovata")
                st.write("_Scarpa sx: ruote 1s–4s | Scarpa dx: ruote 1d–4d_")

                wheel_number_map = {w: f"Ruota {i+1}" for i, w in enumerate(predictions.keys())}
                for i, (wheel, flip) in enumerate(best_assignment):
                    pos = position_keys[i]
                    ruota_label = wheel_number_map.get(wheel, wheel)
                    flip_status = "FLIPPATA" if flip else "NORMALE"
                    color = "red" if flip else "green"
                    st.markdown(f"<b>{pos}</b> → <b>{ruota_label}</b> "
                                f"<span style='color:{color}; font-weight:bold;'>{flip_status}</span>",
                                unsafe_allow_html=True)

                # --- VISUALIZZAZIONE GRAFICA FINALE ---
                left_positions, right_positions = [], []
                for idx, (wheel, flip) in enumerate(best_assignment):
                    pos = position_keys[idx]
                    status = "FLIPPATA" if flip else "NORMALE"
                    wL, wR = predictions[wheel]
                    wear_left, wear_right = (wL, wR) if flip == 0 else (wR, wL)
                    ruota_label = wheel_number_map.get(wheel, wheel)
                    if pos.endswith("s"):
                        left_positions.append((pos, ruota_label, wear_left, wear_right, status))
                    else:
                        right_positions.append((pos, ruota_label, wear_left, wear_right, status))

                left_positions.sort(key=lambda x: int(x[0][0]))
                right_positions.sort(key=lambda x: int(x[0][0]))

                fig, ax = plt.subplots(figsize=(12,6))
                ax.set_xlim(0,10)
                ax.set_ylim(0,6)
                ax.axis('off')

                fig.patch.set_facecolor("#E8F6F5")
                ax.set_facecolor("#E8F6F5")
                ax.axvline(x=5, color="#15B8A6", linestyle="--", linewidth=2, alpha=0.7)

                ax.text(3.3, 6.2, "Scarpa sx", ha='center', va='bottom',
                        fontsize=13, fontweight='bold', color="#0c7b75")
                ax.text(7.2, 6.2, "Scarpa dx", ha='center', va='bottom',
                        fontsize=13, fontweight='bold', color="#0c7b75")

                ruota_radius_x, ruota_radius_y = 0.22, 0.65
                x_left_base, x_right_base = 2.8, 7.2
                y_positions = [5.0, 3.8, 2.6, 1.4]

                def draw_wheel(ax, x, y, label, L, R, status):
                    color = "#90EE90" if status == "NORMALE" else "#FF9999"
                    ellipse = mpatches.Ellipse((x,y),
                                               width=ruota_radius_x*2, height=ruota_radius_y*2,
                                               facecolor=color, edgecolor="black", linewidth=1.5)
                    ax.add_patch(ellipse)
                    x_text = x + (ruota_radius_x + 0.3)
                    line_height = 0.3
                    ax.text(x_text, y + line_height, f"{label}",
                            ha='left', va='center', fontsize=9, fontweight='bold')
                    ax.text(x_text, y, f"L:{L:.2f}", ha='left', va='center',
                            fontsize=8, color='gray')
                    ax.text(x_text, y - line_height, f"R:{R:.2f}", ha='left', va='center',
                            fontsize=8, color='gray')

                for i, (_, label, L, R, s) in enumerate(left_positions):
                    draw_wheel(ax, x_left_base, y_positions[i], label, L, R, s)
                for i, (_, label, L, R, s) in enumerate(right_positions):
                    draw_wheel(ax, x_right_base, y_positions[i], label, L, R, s)

                st.pyplot(fig)
                st.success("✅ Ottimizzazione completata!")

# 💬 ASSISTENTE VIRTUALE – ROLLERS BOT con Context Booster
# ----------------------------------------------------------
st.markdown("---")
st.subheader("🤖 Assistente Virtuale – Rollers Bot")

# Inizializza cronologia chat nella sessione Streamlit
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Mostra conversazione precedente
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input utente (componente chat nativo)
if prompt := st.chat_input("Scrivi un messaggio per Rollers Bot..."):
    # ------------------------------------------------------
    # COSTRUZIONE DEL CONTEX BOOSTER
    # ------------------------------------------------------
    context_parts = []

    # ➊ Recupera dati dalla sessione
    predictions = st.session_state.get("predictions")
    best_assignment = st.session_state.get("best_assignment")
    wheel_number_map = st.session_state.get("wheel_number_map")
    position_keys = st.session_state.get("position_keys")

    # ➋ Costruisci il context technical se esiste qualcosa
    if predictions:
        media_usura = {k: round(float(np.mean(v)), 3) for k, v in predictions.items()}
        context_parts.append(f"Valori medi di usura ruote (0=ottima,1=pessima): {media_usura}")

    if best_assignment and wheel_number_map:
        disposition = [
            f"{pos} → {wheel_number_map.get(wheel, wheel)} "
            f"({'FLIPPATA' if flip else 'NORMALE'})"
            for pos, (wheel, flip) in zip(position_keys, best_assignment)
        ]
        context_parts.append("Migliore disposizione trovata:\n" + "\n".join(disposition))

    context = "\n\n".join(context_parts) if context_parts else None

    # ------------------------------------------------------
    # VISUALIZZA INPUT E RICHIESTA AL MODELLO
    # ------------------------------------------------------
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Rollers Bot sta rispondendo..."):
            # Usa la funzione ask_llm dal modulo chat_agent.py
            answer = ask_llm(prompt, context=context)
            st.markdown(answer)

    # Salva nella sessione la conversazione
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Pulsante per pulire la conversazione
if st.button("🗑️ Reset Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()
    