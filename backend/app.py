from flask import Flask, request, jsonify, send_from_directory
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define folders for uploads and diffs
UPLOAD_FOLDER = 'uploads'
DIFF_FOLDER = 'diffs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DIFF_FOLDER, exist_ok=True)

@app.route('/compare', methods=['POST'])
def compare_images():
    try:
        # Get files from request
        file1 = request.files.get('image1')
        file2 = request.files.get('image2')

        if not file1 or not file2:
            return jsonify({"error": "Both image1 and image2 must be provided"}), 400
        
        # Save uploaded images
        img1_path = os.path.join(UPLOAD_FOLDER, file1.filename)
        img2_path = os.path.join(UPLOAD_FOLDER, file2.filename)
        file1.save(img1_path)
        file2.save(img2_path)

        # Read images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            return jsonify({"error": "Uploaded files must be valid images"}), 400

        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Resize images to the same size
        gray1 = cv2.resize(gray1, (gray2.shape[1], gray2.shape[0]))

        # Compute SSIM
        (score, diff) = compare_ssim(gray1, gray2, full=True)
        diff = (diff * 255).astype("uint8")

        # Find contours to highlight differences
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Save diff image
        diff_filename = 'diff.png'
        diff_path = os.path.join(DIFF_FOLDER, diff_filename)
        cv2.imwrite(diff_path, img1)

        # Respond with similarity score and diff image URL
        diff_image_url = f"http://127.0.0.1:5000/diffs/{diff_filename}"
        return jsonify({"similarity": round(score, 2), "diff_image_url": diff_image_url})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/diffs/<filename>', methods=['GET'])
def serve_diff_image(filename):
    return send_from_directory(DIFF_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
