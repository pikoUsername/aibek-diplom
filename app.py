from flask import Flask
from config import Config
from db.models import db, init_db
from routes.error_handlers import register_error_handlers
from routes.export_bp import export_bp
from services.filters import escapejs_filter
from services.login_manager import init_login_manager
from services.mail_manager import init_mail


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.secret_key = "sos"

    app.add_template_filter(escapejs_filter, "escapejs")

    from routes.index import bp as index_bp
    from routes.train import bp as train_bp
    from routes.predict import bp as predict_bp
    from routes.metrics import bp as metrics_bp
    from routes.history import bp as history_bp
    from routes.upload_csv import bp as upload_bp
    from routes.view_data import bp as data_bp
    from routes.file_manager import bp as file_manager_bp
    from routes.anomalies import bp as anomalies_bp
    from routes.trends import bp as trends_bp
    from routes.sku_train import bp as sku_train_bp
    from routes.sku_predict import bp as sku_predict_bp
    from routes.exog_train import bp as exog_train_bp
    from routes.exog_predict import bp as exog_predict_bp
    from routes.returns_train import bp as returns_train_bp
    from routes.returns_predict import bp as returns_predict_bp
    from routes.transform import bp as transfer_data
    from routes.auth import auth_bp
    from routes.compare_models import bp as compare_bp
    from routes.generate_report import bp as report_bp

    register_error_handlers(app)
    app.register_blueprint(index_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(transfer_data)
    app.register_blueprint(data_bp)
    app.register_blueprint(train_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(metrics_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(file_manager_bp)
    app.register_blueprint(anomalies_bp)
    app.register_blueprint(trends_bp)
    app.register_blueprint(sku_train_bp)
    app.register_blueprint(sku_predict_bp)
    app.register_blueprint(exog_train_bp)
    app.register_blueprint(exog_predict_bp)
    app.register_blueprint(returns_train_bp)
    app.register_blueprint(returns_predict_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(compare_bp)
    app.register_blueprint(report_bp)

    init_db(app)
    init_mail(app)
    init_login_manager(app)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
