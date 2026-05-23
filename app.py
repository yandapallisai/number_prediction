from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os

# ============================================
# FLASK APP
# ============================================

app = Flask(__name__)

CORS(app)

# ============================================
# LOAD ONNX MODEL
# ============================================

print("Loading ONNX model...")

model = YOLO("static/best.onnx", task="detect")
print("Model loaded successfully")

# ============================================
# HOME ROUTE
# ============================================

@app.route("/")
def home():

    print("GET / requested")

    return render_template("index.html")

# ============================================
# PREDICTION ROUTE
# ============================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ============================================
        # CHECK IMAGE
        # ============================================

        if "image" not in request.files:

            return jsonify({

                "success": False,

                "error": "No image received"
            })

        # ============================================
        # READ IMAGE
        # ============================================

        file = request.files["image"]

        image_bytes = file.read()

        npimg = np.frombuffer(

            image_bytes,

            np.uint8
        )

        img = cv2.imdecode(

            npimg,

            cv2.IMREAD_COLOR
        )

        # ============================================
        # VALID IMAGE
        # ============================================

        if img is None:

            return jsonify({

                "success": False,

                "error": "Invalid image"
            })

        # ============================================
        # RESIZE IMAGE
        # ============================================

        img = cv2.resize(

            img,

            (640, 640)
        )

        # ============================================
        # RUN YOLO
        # ============================================

        results = model(

            img,

            conf=0.5,

            imgsz=640
        )

        # ============================================
        # STORE DETECTIONS
        # ============================================

        detected = []

        # ============================================
        # PROCESS RESULTS
        # ============================================

        for r in results:

            for box in r.boxes:

                # ============================================
                # CLASS
                # ============================================

                cls = int(box.cls[0])

                label = model.names[cls]

                # Ignore screen
                if label == "screen":

                    continue

                # ============================================
                # CONFIDENCE
                # ============================================

                conf = float(box.conf[0])

                # ============================================
                # BOX
                # ============================================

                x1, y1, x2, y2 = map(

                    int,

                    box.xyxy[0]
                )

                # ============================================
                # CENTER X
                # ============================================

                center_x = (x1 + x2) / 2

                # ============================================
                # SAVE DETECTION
                # ============================================

                detected.append(

                    (center_x, label)
                )

                # ============================================
                # DRAW RECTANGLE
                # ============================================

                cv2.rectangle(

                    img,

                    (x1, y1),

                    (x2, y2),

                    (0, 255, 0),

                    2
                )

                # ============================================
                # DRAW LABEL
                # ============================================

                cv2.putText(

                    img,

                    f"{label} {conf:.2f}",

                    (x1, y1 - 10),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (0, 255, 0),

                    2
                )

                print(

                    f"Detected: {label}  Confidence: {conf:.2f}"
                )

        # ============================================
        # SORT LEFT TO RIGHT
        # ============================================

        detected.sort(

            key=lambda x: x[0]
        )

        # ============================================
        # REMOVE DUPLICATES
        # ============================================

        filtered = []

        for item in detected:

            exists = False

            for f in filtered:

                if abs(f[0] - item[0]) < 10:

                    exists = True

                    break

            if not exists:

                filtered.append(item)

        # ============================================
        # BUILD FINAL NUMBER
        # ============================================

        final_number = ""

        for item in filtered:

            final_number += str(item[1])

        # ============================================
        # PRINT RESULT
        # ============================================

        print("\n==========================")

        print("FINAL DETECTED NUMBER:")

        print(final_number)

        print("==========================\n")

        # ============================================
        # SAVE OUTPUT
        # ============================================

        cv2.imwrite(

            "output.png",

            img
        )

        # ============================================
        # RETURN RESPONSE
        # ============================================

        return jsonify({

            "success": True,

            "number": final_number
        })

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({

            "success": False,

            "error": str(e)
        })

# ============================================
# RUN APP
# ============================================

if __name__ == "__main__":

    port = int(

        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=True
    )