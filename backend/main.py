"""
AI Skin Analysis Demo - FastAPI Backend

Requirements:
    Python 3.12 (recommended) or Python 3.10+

Installation:
    Basic: pip install fastapi uvicorn opencv-python pillow numpy python-multipart
    With MTCNN (best accuracy): pip install -r requirements.txt
    
    The system will automatically use the best available detector:
    1. MTCNN (if installed) - Highest accuracy (98%), requires TensorFlow 2.13+
    2. face-recognition (if installed) - High accuracy (95%)
    3. OpenCV DNN (auto-downloads model) - Good accuracy (85%)
    4. Haar Cascade (built-in) - Basic accuracy (70%)

Run:
    uvicorn main:app --reload

The server will start at http://localhost:8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
from typing import Dict
import os
import base64

# Face detection setup - try multiple methods in order of preference
USE_MTCNN = False
USE_FACE_RECOGNITION = False
mtcnn_detector = None
dnn_net = None
face_cascade = None
MODEL_NAME = "Unknown"
MODEL_ACCURACY = "Unknown"

# Try MTCNN first (highest accuracy, requires TensorFlow)
try:
    from mtcnn import MTCNN
    mtcnn_detector = MTCNN()
    USE_MTCNN = True
    MODEL_NAME = "MTCNN (Multi-task CNN)"
    MODEL_ACCURACY = "98%"
    print("✓ Using MTCNN face detector (highest accuracy)")
except ImportError:
    print("MTCNN not available, trying face-recognition...")
    # Try face-recognition library (best accuracy, no TensorFlow needed)
    try:
        import face_recognition
        USE_FACE_RECOGNITION = True
        MODEL_NAME = "face-recognition (dlib HOG)"
        MODEL_ACCURACY = "95%"
        print("✓ Using face-recognition library (high accuracy, dlib-based)")
    except ImportError:
        print("face-recognition not available, trying OpenCV DNN...")
        # Try OpenCV DNN face detector (better than Haar Cascade)
        try:
            import urllib.request
            import ssl
            
            # Create SSL context that doesn't verify certificates (for model download)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            
            prototxt_path = "deploy.prototxt"
            model_path = "res10_300x300_ssd_iter_140000.caffemodel"
            
            # Download model files if they don't exist
            def download_file(url, filepath):
                """Download file with SSL context handling"""
                try:
                    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
                    urllib.request.install_opener(opener)
                    with urllib.request.urlopen(url, context=ssl_context) as response:
                        with open(filepath, 'wb') as out_file:
                            out_file.write(response.read())
                    return True
                except Exception as e:
                    print(f"Download failed: {e}")
                    return False
            
            if not os.path.exists(prototxt_path):
                print("Downloading OpenCV DNN prototxt file...")
                if not download_file(prototxt_url, prototxt_path):
                    raise Exception("Could not download prototxt file")
            
            if not os.path.exists(model_path):
                print("Downloading OpenCV DNN model file (this may take a minute)...")
                if not download_file(model_url, model_path):
                    raise Exception("Could not download model file")
            
            if os.path.exists(prototxt_path) and os.path.exists(model_path):
                dnn_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                MODEL_NAME = "OpenCV DNN (ResNet-10 SSD)"
                MODEL_ACCURACY = "85%"
                print("✓ Using OpenCV DNN face detector (good accuracy)")
            else:
                raise Exception("Could not download DNN model files")
        except Exception as e:
            print(f"OpenCV DNN setup failed: {e}")
            # Final fallback to Haar Cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            MODEL_NAME = "OpenCV Haar Cascade"
            MODEL_ACCURACY = "70%"
            print("⚠ Using OpenCV Haar Cascade (basic accuracy)")

app = FastAPI(title="AI Skin Analysis API")

# Enable CORS for localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_image(file: UploadFile) -> None:
    """Validate uploaded image file."""
    # Check file extension
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPG, JPEG, and PNG are allowed."
        )
    
    # File size validation will be handled by FastAPI's size limits


def detect_face(image: np.ndarray) -> tuple:
    """
    Detect face in image using MTCNN (preferred), face-recognition, OpenCV DNN, or Haar Cascade fallback.
    Returns: (face_detected: bool, face_bbox: tuple or None)
    """
    h, w = image.shape[:2]
    
    if USE_MTCNN and mtcnn_detector is not None:
        # Use MTCNN (Multi-task Cascaded Convolutional Networks) - highest accuracy
        # MTCNN expects RGB format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = mtcnn_detector.detect_faces(rgb_image)
        
        if not faces or len(faces) == 0:
            return False, None
        
        # Get the first (and most confident) face
        # MTCNN returns list of dicts with 'box' key: [x, y, width, height]
        # Sort by confidence if available, otherwise by size
        if 'confidence' in faces[0]:
            faces = sorted(faces, key=lambda f: f.get('confidence', 0), reverse=True)
        else:
            faces = sorted(faces, key=lambda f: f['box'][2] * f['box'][3], reverse=True)
        
        face_box = faces[0]['box']
        x, y, width, height = face_box
        
        # Ensure coordinates are within image bounds
        x = max(0, int(x))
        y = max(0, int(y))
        width = min(int(width), w - x)
        height = min(int(height), h - y)
        
        # Ensure we have valid dimensions
        if width <= 0 or height <= 0:
            return False, None
        
        return True, (x, y, width, height)
    
    elif USE_FACE_RECOGNITION:
        # Use face-recognition library (dlib-based, very accurate)
        # face_recognition expects RGB format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image, model='hog')
        
        if not face_locations or len(face_locations) == 0:
            return False, None
        
        # Get the first (and largest) face
        # face_locations returns (top, right, bottom, left)
        top, right, bottom, left = face_locations[0]
        
        x = left
        y = top
        width = right - left
        height = bottom - top
        
        # Ensure coordinates are within image bounds
        x = max(0, x)
        y = max(0, y)
        width = min(width, w - x)
        height = min(height, h - y)
        
        # Ensure we have valid dimensions
        if width <= 0 or height <= 0:
            return False, None
        
        return True, (x, y, width, height)
    
    elif dnn_net is not None:
        # Use OpenCV DNN face detector
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 
            1.0, 
            (300, 300), 
            [104, 117, 123]
        )
        dnn_net.setInput(blob)
        detections = dnn_net.forward()
        
        # Find the best detection
        best_confidence = 0.5
        best_face = None
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > best_confidence:
                best_confidence = confidence
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                best_face = (x1, y1, x2 - x1, y2 - y1)
        
        if best_face:
            x, y, width, height = best_face
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)
            if width > 0 and height > 0:
                return True, (x, y, width, height)
        
        return False, None
    
    else:
        # Fallback to OpenCV Haar Cascade with improved parameters
        if face_cascade is None:
            return False, None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try multiple detection strategies for better results
        # Strategy 1: Standard detection
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Strategy 2: If no faces found, try more sensitive parameters
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,  # Smaller steps
                minNeighbors=3,    # Lower threshold
                minSize=(20, 20),  # Smaller minimum size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        # Strategy 3: Try with different scale
        if len(faces) == 0:
            # Equalize histogram for better contrast
            equalized = cv2.equalizeHist(gray)
            faces = face_cascade.detectMultiScale(
                equalized,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(40, 40),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        if len(faces) == 0:
            return False, None
        
        # Get the largest face (most likely to be the main subject)
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, width, height = faces[0]
        
        # Expand face region slightly for better skin analysis
        expand_factor = 0.1
        x = max(0, int(x - width * expand_factor))
        y = max(0, int(y - height * expand_factor))
        width = min(int(width * (1 + 2 * expand_factor)), w - x)
        height = min(int(height * (1 + 2 * expand_factor)), h - y)
        
        return True, (x, y, width, height)


def analyze_acne(face_roi: np.ndarray) -> float:
    """
    Analyze acne/pimples in face region.
    Uses texture analysis and color variation detection.
    Returns probability score (0-1).
    """
    # Convert to grayscale for texture analysis
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to smooth skin
    smooth = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Calculate difference to detect irregularities
    diff = cv2.absdiff(gray, smooth)
    
    # Threshold to find significant variations
    _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
    
    # Calculate percentage of irregularities
    irregularity_ratio = np.sum(thresh > 0) / (thresh.shape[0] * thresh.shape[1])
    
    # Analyze redness in the face region (acne often appears red)
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    # Red color range in HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    redness_ratio = np.sum(red_mask > 0) / (red_mask.shape[0] * red_mask.shape[1])
    
    # Combine irregularity and redness metrics
    acne_score = min(1.0, (irregularity_ratio * 2.0 + redness_ratio * 1.5) / 2.0)
    
    return float(acne_score)


def analyze_dark_circles(face_roi: np.ndarray, face_bbox: tuple, image_shape: tuple) -> float:
    """
    Analyze dark circles under eyes.
    Focuses on the lower eye region.
    Returns probability score (0-1).
    """
    h, w = image_shape[:2]
    fx, fy, fw, fh = face_bbox
    
    # Estimate eye region (typically in upper-middle portion of face)
    # Under-eye area is roughly 20-40% down from top of face, and 20-80% across
    eye_y_start = int(fy + fh * 0.25)
    eye_y_end = int(fy + fh * 0.45)
    eye_x_start = int(fx + fw * 0.2)
    eye_x_end = int(fx + fw * 0.8)
    
    # Ensure coordinates are within bounds
    eye_y_start = max(0, eye_y_start)
    eye_y_end = min(h, eye_y_end)
    eye_x_start = max(0, eye_x_start)
    eye_x_end = min(w, eye_x_end)
    
    if eye_y_end <= eye_y_start or eye_x_end <= eye_x_start:
        return 0.0
    
    eye_region = face_roi[
        eye_y_start - fy:eye_y_end - fy,
        eye_x_start - fx:eye_x_end - fx
    ]
    
    if eye_region.size == 0:
        return 0.0
    
    # Convert to grayscale
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    
    # Calculate average brightness (darker = lower values)
    avg_brightness = np.mean(gray_eye)
    
    # Normalize: lower brightness = higher dark circle probability
    # Typical skin brightness is around 120-180, darker circles are 60-100
    if avg_brightness < 80:
        dark_circle_score = 0.8 + (80 - avg_brightness) / 80 * 0.2
    elif avg_brightness < 120:
        dark_circle_score = 0.3 + (120 - avg_brightness) / 40 * 0.5
    else:
        dark_circle_score = max(0.0, (140 - avg_brightness) / 20 * 0.3)
    
    return float(min(1.0, max(0.0, dark_circle_score)))


def analyze_dark_spots(face_roi: np.ndarray) -> float:
    """
    Analyze dark spots/hyperpigmentation.
    Detects darker patches in the face region.
    Returns probability score (0-1).
    """
    # Convert to LAB color space for better skin tone analysis
    lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]  # Lightness channel
    
    # Calculate mean and standard deviation
    mean_lightness = np.mean(l_channel)
    std_lightness = np.std(l_channel)
    
    # Dark spots are regions significantly darker than average
    threshold = mean_lightness - (std_lightness * 1.5)
    
    # Create mask for dark regions
    dark_mask = l_channel < threshold
    
    # Calculate percentage of dark spots
    dark_spot_ratio = np.sum(dark_mask) / (l_channel.shape[0] * l_channel.shape[1])
    
    # Also check for color variation (hyperpigmentation shows as brown/dark patches)
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    # Brown/dark color range
    lower_brown = np.array([10, 50, 20])
    upper_brown = np.array([30, 255, 150])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    brown_ratio = np.sum(brown_mask > 0) / (brown_mask.shape[0] * brown_mask.shape[1])
    
    # Combine metrics
    dark_spot_score = min(1.0, (dark_spot_ratio * 3.0 + brown_ratio * 2.0) / 2.5)
    
    return float(dark_spot_score)


def draw_face_box(image: np.ndarray, face_bbox: tuple) -> np.ndarray:
    """
    Draw a bounding box around the detected face.
    Returns: Image with bounding box drawn
    """
    x, y, w, h = face_bbox
    # Create a copy to avoid modifying the original
    marked_image = image.copy()
    
    # Draw rectangle (BGR format: green color)
    color = (0, 255, 0)  # Green in BGR
    thickness = 3
    
    # Draw the main bounding box
    cv2.rectangle(marked_image, (x, y), (x + w, y + h), color, thickness)
    
    # Add corner markers for better visibility
    corner_length = min(w, h) // 8
    # Top-left corner
    cv2.line(marked_image, (x, y), (x + corner_length, y), color, thickness)
    cv2.line(marked_image, (x, y), (x, y + corner_length), color, thickness)
    # Top-right corner
    cv2.line(marked_image, (x + w, y), (x + w - corner_length, y), color, thickness)
    cv2.line(marked_image, (x + w, y), (x + w, y + corner_length), color, thickness)
    # Bottom-left corner
    cv2.line(marked_image, (x, y + h), (x + corner_length, y + h), color, thickness)
    cv2.line(marked_image, (x, y + h), (x, y + h - corner_length), color, thickness)
    # Bottom-right corner
    cv2.line(marked_image, (x + w, y + h), (x + w - corner_length, y + h), color, thickness)
    cv2.line(marked_image, (x + w, y + h), (x + w, y + h - corner_length), color, thickness)
    
    # Add label
    label = "Face Detected"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    # Draw label background
    label_y = max(y - 10, text_height + 10)
    cv2.rectangle(marked_image, 
                  (x, label_y - text_height - 5), 
                  (x + text_width + 10, label_y + 5), 
                  color, -1)
    
    # Draw label text
    cv2.putText(marked_image, label, (x + 5, label_y), 
                font, font_scale, (0, 0, 0), font_thickness)
    
    return marked_image


def image_to_base64(image: np.ndarray) -> str:
    """
    Convert OpenCV image (BGR) to base64 encoded string (JPEG format).
    Returns: Base64 encoded image string
    """
    # Convert BGR to RGB for PIL
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    # Convert to bytes
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=85)
    image_bytes = buffer.getvalue()
    
    # Encode to base64
    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_string}"


def generate_recommendations(acne_score: float, dark_circles_score: float, dark_spots_score: float) -> Dict:
    """
    Generate personalized skin care recommendations based on analysis scores.
    Returns a dictionary with recommendations and priority areas.
    """
    threshold = 0.5  # Score above which condition needs attention
    recommendations = []
    priority_areas = []
    
    # Acne recommendations
    if acne_score >= threshold:
        priority_areas.append("Acne/Pimples")
        recommendations.append({
            "condition": "Acne / Pimples",
            "severity": "High" if acne_score >= 0.7 else "Moderate",
            "score": acne_score,
            "tips": [
                "Use a gentle, non-comedogenic cleanser twice daily",
                "Apply salicylic acid or benzoyl peroxide spot treatment",
                "Avoid touching your face with unwashed hands",
                "Change pillowcases regularly (every 2-3 days)",
                "Consider consulting a dermatologist for persistent acne"
            ]
        })
    
    # Dark Circles recommendations
    if dark_circles_score >= threshold:
        priority_areas.append("Dark Circles")
        recommendations.append({
            "condition": "Dark Circles",
            "severity": "High" if dark_circles_score >= 0.7 else "Moderate",
            "score": dark_circles_score,
            "tips": [
                "Get 7-9 hours of quality sleep each night",
                "Apply a cold compress or cucumber slices to reduce puffiness",
                "Use an eye cream with vitamin C, retinol, or caffeine",
                "Stay hydrated and maintain a balanced diet",
                "Protect under-eye area from sun exposure with SPF"
            ]
        })
    
    # Dark Spots recommendations
    if dark_spots_score >= threshold:
        priority_areas.append("Dark Spots")
        recommendations.append({
            "condition": "Dark Spots / Hyperpigmentation",
            "severity": "High" if dark_spots_score >= 0.7 else "Moderate",
            "score": dark_spots_score,
            "tips": [
                "Use sunscreen daily (SPF 30+) to prevent further darkening",
                "Apply products with vitamin C, niacinamide, or alpha arbutin",
                "Consider chemical exfoliants (AHA/BHA) 2-3 times per week",
                "Avoid picking at spots to prevent post-inflammatory hyperpigmentation",
                "Be patient - dark spots can take 3-6 months to fade"
            ]
        })
    
    # Overall skin health message
    if len(recommendations) == 0:
        overall_message = "Great! Your skin looks healthy. Continue your current skincare routine."
    elif len(recommendations) == 1:
        overall_message = f"Focus on improving {priority_areas[0].lower()} for better skin health."
    else:
        overall_message = f"Priority areas to address: {', '.join(priority_areas[:-1])} and {priority_areas[-1]}."
    
    return {
        "overallMessage": overall_message,
        "recommendations": recommendations,
        "priorityCount": len(recommendations)
    }


@app.post("/analyze")
async def analyze_skin(image: UploadFile = File(...)):
    """
    Analyze skin conditions in uploaded face image.
    
    Accepts: multipart/form-data with field name "image"
    Returns: JSON with acne, darkCircles, and darkSpots probabilities
    """
    try:
        # Validate image
        validate_image(image)
        
        # Read image file
        contents = await image.read()
        
        # Check file size (5MB limit)
        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File size exceeds 5MB limit."
            )
        
        # Convert to OpenCV format
        image_bytes = io.BytesIO(contents)
        pil_image = Image.open(image_bytes)
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Detect face
        face_detected, face_bbox = detect_face(opencv_image)
        
        if not face_detected:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "message": "No face detected"
                }
            )
        
        # Extract face region
        x, y, w, h = face_bbox
        face_roi = opencv_image[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "message": "Invalid face region"
                }
            )
        
        # Analyze skin conditions
        acne_score = analyze_acne(face_roi)
        dark_circles_score = analyze_dark_circles(face_roi, face_bbox, opencv_image.shape)
        dark_spots_score = analyze_dark_spots(face_roi)
        
        # Round to 2 decimal places
        acne_score = round(acne_score, 2)
        dark_circles_score = round(dark_circles_score, 2)
        dark_spots_score = round(dark_spots_score, 2)
        
        # Generate recommendations
        recommendations_data = generate_recommendations(acne_score, dark_circles_score, dark_spots_score)
        
        # Draw bounding box on the detected face
        marked_image = draw_face_box(opencv_image, face_bbox)
        
        # Convert marked image to base64
        marked_image_base64 = image_to_base64(marked_image)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "results": {
                    "acne": acne_score,
                    "darkCircles": dark_circles_score,
                    "darkSpots": dark_spots_score
                },
                "recommendations": recommendations_data,
                "modelInfo": {
                    "name": MODEL_NAME,
                    "accuracy": MODEL_ACCURACY
                },
                "markedImage": marked_image_base64
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "AI Skin Analysis API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

