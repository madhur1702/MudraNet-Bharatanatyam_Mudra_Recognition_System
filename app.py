import streamlit as st
import numpy as np
import cv2
import json
import mediapipe as mp
from tensorflow.keras.models import load_model

@st.cache_resource
def load_assets():
    model = load_model("mudranet_final.keras")
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_assets()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

@st.cache_resource
def load_landmarker():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3
    )
    return HandLandmarker.create_from_options(options)

landmarker = load_landmarker()

mudra_meanings = {
    "Pathaka(1)": "Flag - used to depict sky, forest, or waving",
    "Tripathaka(1)": "Three part flag - used for crown or tree",
    "Ardhapathaka(1)": "Half flag - used for knife or tower",
    "Mayura(1)": "Peacock - used to depict a peacock or stroking hair",
    "Anjali(1)": "Salutation - used for greeting or prayer",
    "Katakamukha_1": "Opening in a link - used for holding a garland",
    "Katakamukha_2": "Katakamukha variant 2 - used for picking flowers",
    "Katakamukha_3": "Katakamukha variant 3 - used for delicate gestures",
    "Alapadmam(1)": "Full blown lotus - used for beauty or moon",
    "Mushti(1)": "Fist - used for holding or strength",
    "Sikharam(1)": "Spire - used for bow or pillar",
    "Kapith(1)": "Wood apple - used for goddess Lakshmi",
    "Suchi(1)": "Needle - used for pointing or the world",
    "Chandrakala(1)": "Moon digit - used for moon or forehead mark",
    "Padmakosha(1)": "Lotus bud - used for fruits or a ball",
    "Sarpasirsha(1)": "Snake head - used for elephant or snake",
    "Mrigasirsha(1)": "Deer head - used for deer or woman face",
    "Simhamukham(1)": "Lion face - used for lion or valour",
    "Kangulam(1)": "Used for fruits or hanging objects",
    "Mukulam(1)": "Bud - used for eating or lotus bud",
    "Aralam(1)": "Curved - used for wind god or drinking nectar",
    "Shukatundam(1)": "Parrot beak - used for parrot or arrow",
    "Bramaram(1)": "Bee - used for bee or fighting",
    "Hamsasyam(1)": "Swan face - used for painting or fine work",
    "Hamsapaksham(1)": "Swan wing - used for swan or moonlight",
    "Tamarachudam(1)": "Cock - used for cock or peacock feather",
    "Trishulam(1)": "Trident - used for Lord Shiva trident",
    "Katrimukha(1)": "Scissors face - used for opposition or separation",
    "Ardhachandran(1)": "Half moon - used for moon or blessing",
    "Chaturam(1)": "Clever - used for gold or clever person",
    "Swastikam(1)": "Auspicious symbol - used for sacred occasions",
    "Kapotham(1)": "Dove - used for respectful salutation",
    "Karkatta(1)": "Crab - used for crab or interlocking fingers",
    "Pushpaputa(1)": "Handful of flowers - used for offering flowers",
    "Shivalinga(1)": "Symbol of Lord Shiva",
    "Katakavardhana(1)": "Used for coronation ceremonies",
    "Kartariswastika(1)": "Scissors crossed - used for opposition",
    "Sakata(1)": "Carriage - used for demon or carriage",
    "Shanka(1)": "Conch shell - used for conch or pure being",
    "Chakra(1)": "Wheel - used for wheel or discus",
    "Samputa(1)": "Covered box - used for box or secret",
    "Pasha(1)": "Noose - used for bond or rope",
    "Kilaka(1)": "Bond - used for friendship or bonding",
    "Matsya(1)": "Fish - used to depict a fish",
    "Kurma(1)": "Tortoise - used to depict a tortoise",
    "Varaha(1)": "Boar - used for boar avatar of Vishnu",
    "Garuda(1)": "Eagle - used for Garuda or bird",
    "Nagabandha(1)": "Snake bond - used for intertwined snakes",
    "Khatva(1)": "Cot - used for cot or bedstead",
    "Berunda(1)": "Two headed bird - used for power",
}

def extract_landmarks(img_rgb):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = landmarker.detect(mp_image)
    if results.hand_landmarks:
        lm_array = []
        for hand in results.hand_landmarks:
            for lm in hand:
                lm_array.extend([lm.x, lm.y, lm.z])
        lm_array = lm_array[:126]
        lm_array += [0.0] * (126 - len(lm_array))
        return np.array(lm_array, dtype="float32"), results
    return None, results

def draw_landmarks(img_rgb, results):
    img_display = img_rgb.copy()
    if results.hand_landmarks:
        h, w, _ = img_rgb.shape
        for hand in results.hand_landmarks:
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for pt in points:
                cv2.circle(img_display, pt, 6, (0, 255, 0), -1)
    return img_display

def predict(img_rgb):
    img_resized = cv2.resize(img_rgb, (224, 224)).astype("float32") / 255.0
    lm_array, results = extract_landmarks(img_rgb)
    if lm_array is not None:
        img_input = np.expand_dims(img_resized, axis=0)
        lm_input = np.expand_dims(lm_array, axis=0)
        preds = model.predict({"image_input": img_input, "landmark_input": lm_input}, verbose=0)
        top3_idx = np.argsort(preds[0])[::-1][:3]
        return preds[0], top3_idx, results
    return None, None, results

st.set_page_config(page_title="MudraNet", page_icon="🤚", layout="wide")
st.title("🤚 MudraNet: Bharatiya Natya Mudra Identifier")
st.markdown("Identify Bharatanatyam hand gestures using your laptop camera or by uploading an image.")

mode = st.radio("Choose input mode", ["📷 Live Camera", "🖼️ Upload Image"], horizontal=True)

if mode == "📷 Live Camera":
    st.info("Tick the checkbox to start camera. Your hand gesture will be predicted in real time.")
    run = st.checkbox("▶️ Start Camera")
    FRAME_WINDOW = st.empty()
    result_placeholder = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access camera. Make sure no other app is using it.")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from camera.")
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preds, top3_idx, results = predict(frame_rgb)
                frame_lm = draw_landmarks(frame_rgb, results)
                FRAME_WINDOW.image(frame_lm, channels="RGB", use_container_width=True)
                if preds is not None:
                    predicted_class = class_names[top3_idx[0]]
                    confidence = preds[top3_idx[0]] * 100
                    meaning = mudra_meanings.get(predicted_class, "Meaning not available")
                    result_placeholder.success(f"**{predicted_class}** — {confidence:.1f}% confidence\n\n📖 {meaning}")
                else:
                    result_placeholder.warning("No hand detected — show your hand clearly to the camera")
            cap.release()

else:
    uploaded_file = st.file_uploader("Upload hand gesture image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        preds, top3_idx, results = predict(img_rgb)
        img_lm = draw_landmarks(img_rgb, results)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Original")
            st.image(img_rgb, use_container_width=True)
        with col2:
            st.subheader("Landmarks")
            st.image(img_lm, use_container_width=True)
        with col3:
            st.subheader("Prediction")
            if preds is not None:
                predicted_class = class_names[top3_idx[0]]
                confidence = preds[top3_idx[0]] * 100
                meaning = mudra_meanings.get(predicted_class, "Meaning not available")
                st.success(f"**{predicted_class}**")
                st.metric("Confidence", f"{confidence:.1f}%")
                st.info(f"📖 {meaning}")
                st.subheader("Top 3 Predictions")
                for idx in top3_idx:
                    name = class_names[idx]
                    prob = preds[idx] * 100
                    st.progress(int(prob), text=f"{name}: {prob:.1f}%")
            else:
                st.error("No hand detected. Try a clearer image.")
