from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np

# Flask app
app = Flask(__name__)
CORS(app)

# Load YOLO model
# Use best.onnx OR best.pt
model = YOLO("best.pt", task="detect")

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Check image exists
        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error": "No image received"
            })

        # Get image
        file = request.files["image"]

        # Convert image to OpenCV format
        image_bytes = file.read()

        npimg = np.frombuffer(image_bytes, np.uint8)

        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Check image valid
        if img is None:

            return jsonify({
                "success": False,
                "error": "Invalid image"
            })

        # Run YOLO inference
        results = model(
            img,
            conf=0.7,      # Higher confidence
            imgsz=1280     # Better for tiny dots
        )

        detected = []

        # Process detections
        for r in results:

            for box in r.boxes:

                # Class ID
                cls = int(box.cls[0])

                # Label name
                label = model.names[cls]

                # Ignore screen class
                if label == "screen":
                    continue

                # Confidence
                conf = float(box.conf[0])

                # Bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Center x for sorting
                center_x = (x1 + x2) / 2

                # Save detection
                detected.append((center_x, label))

                # Draw rectangle
                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Draw label
                cv2.putText(
                    img,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                print(f"Detected: {label}  Confidence: {conf:.2f}")

        # Sort left to right
        detected.sort(key=lambda x: x[0])

        # Build final number
        final_number = ""

        for item in detected:
            final_number += str(item[1])

        print("\n==========================")
        print("FINAL DETECTED NUMBER:", final_number)
        print("==========================\n")

        # Save output image
        cv2.imwrite("output.png", img)

        # Return response
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

# Run Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
