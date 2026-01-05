# AI Skin Analysis Demo

A simple full-stack application for face detection and skin condition analysis using pretrained models.

## Features

- **Face Detection**: Uses MTCNN (preferred) or OpenCV DNN for accurate face detection
- **Skin Condition Analysis**: Detects three conditions:
  - Acne / Pimples
  - Dark Circles under eyes
  - Dark Spots / Hyperpigmentation
- **REST API**: FastAPI backend with JSON responses
- **Simple UI**: Bootstrap-based frontend for easy testing

## Installation

### Requirements

- **Python 3.12** (recommended) or Python 3.10+
- pip package manager

### Backend Dependencies

**1. Create a virtual environment with Python 3.12:**
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**2. Install all dependencies:**
```bash
pip install -r requirements.txt
```

**Basic installation (without MTCNN):**
```bash
pip install fastapi uvicorn opencv-python pillow numpy python-multipart
```

**Note**: The system automatically uses the best available detector:
1. **MTCNN** (if installed) - Highest accuracy (98%), requires TensorFlow 2.13+
2. **face-recognition** (if installed) - High accuracy (95%), uses dlib (no TensorFlow needed)
3. **OpenCV DNN** (auto-downloads model) - Good accuracy (85%), works out of the box
4. **Haar Cascade** (built-in) - Basic accuracy (70%), always available as fallback

### Frontend

No installation needed - uses Bootstrap CDN. Just open `frontend/index.html` in a browser.

## Running the Application

### 1. Start the Backend Server

```bash
cd backend
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### 2. Open the Frontend

Open `frontend/index.html` in your web browser, or serve it using a simple HTTP server:

```bash
# Using Python
cd frontend
python -m http.server 8080

# Then open http://localhost:8080 in your browser
```

## API Endpoints

### POST /analyze

Analyzes skin conditions in an uploaded face image.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Field name: `image`
- Accepted formats: JPG, JPEG, PNG
- Max file size: 5MB

**Success Response:**
```json
{
  "success": true,
  "results": {
    "acne": 0.83,
    "darkCircles": 0.61,
    "darkSpots": 0.42
  }
}
```

**Error Response (No Face):**
```json
{
  "success": false,
  "message": "No face detected"
}
```

## Project Structure

```
FACE DETECTION/
├── backend/
│   └── main.py          # FastAPI application
├── frontend/
│   └── index.html       # Bootstrap UI
└── README.md            # This file
```

## How It Works

1. **Face Detection**: Uses MTCNN (preferred), face-recognition library, OpenCV DNN, or Haar Cascade fallback to locate faces in the image
2. **Region Extraction**: Extracts the face region for analysis
3. **Skin Analysis**: Applies heuristic-based algorithms to detect:
   - **Acne**: Texture irregularities and redness detection
   - **Dark Circles**: Brightness analysis in under-eye region
   - **Dark Spots**: Hyperpigmentation detection using color space analysis

## Notes

- Only processes **one face per image**
- Uses **pretrained models only** (no training required)
- All libraries are **free and open-source**
- Results are probability scores (0-1) indicating likelihood of each condition

## Integration

This demo is designed to be easily integrated into Shopify or other platforms. The backend provides a simple REST API that can be called from any frontend framework.

## Troubleshooting

- **CORS errors**: The backend has CORS enabled for all origins. In production, restrict to specific domains.
- **No face detected**: Ensure the image contains a clear, front-facing face
- **Connection errors**: Make sure the backend is running on `http://localhost:8000`

## License

This project uses open-source libraries. Please refer to individual library licenses.

