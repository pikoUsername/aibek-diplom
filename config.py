import os

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "data")
    MODEL_FOLDER = os.path.join(BASE_DIR, "models")
    PLOT_FOLDER = os.path.join(BASE_DIR, "static", "plots")
    SCRIPT_FOLDER = os.path.join(BASE_DIR, "scripts")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    os.makedirs(PLOT_FOLDER, exist_ok=True)
    os.makedirs(SCRIPT_FOLDER, exist_ok=True)
