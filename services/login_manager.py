from flask_login import LoginManager

from db.models import db, User

login_manager = LoginManager()


def init_login_manager(app):
    login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
