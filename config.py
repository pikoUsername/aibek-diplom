import os

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "data")
    MODEL_FOLDER = os.path.join(BASE_DIR, "models")
    PLOT_FOLDER = os.path.join(BASE_DIR, "static", "plots")
    SCRIPT_FOLDER = os.path.join(BASE_DIR, "scripts")
    SQLALCHEMY_DATABASE_URI = os.environ.get("SQLALCHEMY_DATABASE_URI", "sqlite:///project.db")

    # в config.py
    MAIL_SERVER = os.environ.get("MAIL_SERVER", 'smtp.yandex.ru')
    MAIL_PORT = os.environ.get("MAIL_PORT", 587)
    MAIL_USE_TLS = os.environ.get("MAIL_USE_TLS", True)
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME", 'karma-team228')
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", 'fsxbefwweetutubq')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    os.makedirs(PLOT_FOLDER, exist_ok=True)
    os.makedirs(SCRIPT_FOLDER, exist_ok=True)
