import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2
import numpy as np

# ----------------------------------------------------
# PAGE SETTINGS
# ----------------------------------------------------

st.set_page_config(
    page_title="Smart Retail Shelf Monitoring",
    layout="wide"
)

st.title("🛒 Smart Retail Shelf Monitoring System")

st.write(
    "Upload a supermarket shelf image to detect shelves and estimate shelf occupancy."
)

# ----------------------------------------------------
# LOAD YOLO MODEL
# ----------------------------------------------------

model = YOLO("runs/detect/train/weights/best.pt")

# ----------------------------------------------------
# IMAGE UPLOAD
# ----------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Shelf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    if st.button("🔍 Analyze Shelf"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp:

            image.save(temp.name)

            temp_path = temp.name

        # ----------------------------
        # YOLO Prediction
        # ----------------------------

        results = model.predict(
            source=temp_path,
            conf=0.25,
            save=False
        )

        boxes = results[0].boxes

        original = cv2.imread(temp_path)

        detected = original.copy()

        total_shelves = len(boxes)

        filled = 0
        low = 0
        empty = 0

        shelf_details = []

        # ------------------------------------
        # Analyze Every Shelf
        # ------------------------------------

        for i, box in enumerate(boxes):

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            shelf = original[y1:y2, x1:x2]

            if shelf.size == 0:
                continue

            gray = cv2.cvtColor(
                shelf,
                cv2.COLOR_BGR2GRAY
            )

            # ----------------------------------
            # Better Occupancy Estimation
            # ----------------------------------

            blur = cv2.GaussianBlur(
                gray,
                (5,5),
                0
            )

            thresh = cv2.adaptiveThreshold(
                blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                21,
                6
            )

            kernel = np.ones((3,3),np.uint8)

            thresh = cv2.morphologyEx(
                thresh,
                cv2.MORPH_CLOSE,
                kernel
            )

            white_pixels = np.sum(thresh==255)

            total_pixels = thresh.shape[0]*thresh.shape[1]

            fill = int(
                (white_pixels/total_pixels)*100
            )

            confidence = float(box.conf)
                        # ----------------------------------
            # Shelf Status
            # ----------------------------------

            # These thresholds are much more balanced
            # than the previous edge-based approach.

            if fill >= 35:

                status = "🟢 Filled"
                color = (0, 200, 0)
                filled += 1

            elif fill >= 18:

                status = "🟡 Low Stock"
                color = (0, 255, 255)
                low += 1

            else:

                status = "🔴 Nearly Empty"
                color = (0, 0, 255)
                empty += 1

            shelf_details.append(
                {
                    "id": i + 1,
                    "confidence": confidence,
                    "fill": fill,
                    "status": status
                }
            )

            # -----------------------------
            # Draw Shelf Box
            # -----------------------------

            cv2.rectangle(
                detected,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            cv2.putText(
                detected,
                f"Shelf {i+1}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2
            )

        # ----------------------------------
        # Show Detection Result
        # ----------------------------------

        with col2:

            st.image(
                cv2.cvtColor(
                    detected,
                    cv2.COLOR_BGR2RGB
                ),
                caption="Detection Result",
                use_container_width=True
            )

        st.success(f"✅ Shelves Detected : {total_shelves}")

        # ----------------------------------
        # Dashboard
        # ----------------------------------

        st.subheader("📈 Store Summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Shelves", total_shelves)
        c2.metric("Filled", filled)
        c3.metric("Low Stock", low)
        c4.metric("Nearly Empty", empty)

        if total_shelves > 0:
            utilization = int(((filled + 0.5 * low) / total_shelves) * 100)
        else:
            utilization = 0

        st.metric(
            "Shelf Utilization",
            f"{utilization}%"
        )

        st.progress(utilization / 100)

        if utilization >= 80:

            st.success("✅ Shelf utilization is good.")

        elif utilization >= 50:

            st.warning("⚠️ Some shelves need restocking.")

        else:

            st.error("❌ Shelf utilization is low.")

        # ----------------------------------
        # Shelf Analysis
        # ----------------------------------

        st.subheader("📦 Shelf Analysis")

        for shelf in shelf_details:

            with st.expander(f"Shelf {shelf['id']}"):

                st.write(
                    f"**Detection Confidence:** {shelf['confidence']:.2f}"
                )

                st.progress(
                    shelf["fill"] / 100
                )

                st.write(
                    f"**Estimated Filled:** {shelf['fill']}%"
                )

                st.write(
                    f"**Status:** {shelf['status']}"
                )

        os.remove(temp_path)