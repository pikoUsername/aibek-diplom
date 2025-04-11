from db.models import User

def get_user_by_email(db_session, email):
    """
    Получает пользователя по email.
    """
    return db_session.query(User).filter_by(email=email).first()


def create_user(db_session, username, email, password_hash):
    """
    Создаёт нового пользователя и сохраняет его в базе данных.
    """
    user = User(username=username, email=email, password_hash=password_hash)
    db_session.add(user)
    db_session.commit()
    return user


def get_all_users(db_session):
    """
    Возвращает список всех пользователей.
    """
    return db_session.query(User).all()
