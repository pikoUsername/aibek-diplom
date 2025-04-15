from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func

db = SQLAlchemy()


def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()


class BaseModel(db.Model): 
    __abstract__ = True 

    id = db.Column(db.Integer, primary_key=True, nullable=False)
    created_at = db.Column(db.DateTime, server_default=func.now(), nullable=False)
    updated_at = db.Column(db.DateTime, server_default=func.now(), onupdate=func.current_timestamp(), nullable=False)


class User(UserMixin, BaseModel):
    __tablename__ = 'users'

    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)\

    profile_image = db.Column(db.String(256), nullable=True, default="default.png")  # путь к аватару
    bio = db.Column(db.Text, nullable=True)  # краткая информация о пользователе
    last_login = db.Column(db.DateTime, nullable=True)  # время последнего входа
    social_link = db.Column(db.String(256), nullable=True)  # ссылка на профиль в соцсетях (например, LinkedIn или Facebook)

    role = db.Column(db.String(20), default='user', nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'


class UserAction(BaseModel):
    __tablename__ = 'user_actions'

    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    action_type = db.Column(db.String(50), nullable=False)  # например: "upload", "train", "predict", "transform"
    description = db.Column(db.Text, nullable=True)


class Plot(BaseModel):
    __tablename__ = "plots"

    user_id = db.Column(db.ForeignKey("users.id"), nullable=False)
    plot_filename = db.Column(db.String(256), nullable=False)
