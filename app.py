from flask import Flask
from config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Импорт и регистрация всех роутов
    from routes.index import bp as index_bp
    from routes.upload_csv import bp as upload_bp
    from routes.train import bp as train_bp
    from routes.predict import bp as predict_bp
    from routes.metrics import bp as metrics_bp
    from routes.history import bp as history_bp
    from routes.file_manager import bp as file_manager_bp

    app.register_blueprint(index_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(train_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(metrics_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(file_manager_bp)

    return app


if __name__ == "__main__":
    app = create_app()
