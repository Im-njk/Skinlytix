# Render.com Deployment Guide

## Root Directory Configuration

For deploying the backend to Render.com, use the following settings:

### Root Directory
```
backend
```

### Build Command
```
pip install -r requirements.txt
```

### Start Command
```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Render.com Service Settings

1. **Service Type**: Web Service
2. **Environment**: Python 3
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. **Root Directory**: `backend`

## Important Notes

- The `requirements.txt` file is now in the `backend/` folder
- Model files (`deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`) are in the `backend/` folder
- Render.com will automatically use the `$PORT` environment variable
- Make sure to set Python version to 3.12 in Render.com settings (or 3.10+)

## Environment Variables (Optional)

You can set these in Render.com dashboard if needed:
- `PYTHON_VERSION=3.12` (if not auto-detected)

## CORS Configuration

The backend is already configured to accept requests from any origin. If you want to restrict it to your frontend domain, update the CORS settings in `main.py`.

