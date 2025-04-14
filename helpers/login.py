from flask_login import current_user
from functools import wraps
from flask import abort

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return abort(401)  # или redirect на страницу логина
        if current_user.role != 'admin':
            return abort(403)  # доступ запрещен
        return f(*args, **kwargs)
    return decorated_function
