from flask import Blueprint, request, render_template, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from db.user_repo import get_user_by_email, create_user, get_all_users
from db.models import db, User

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    errors = []
    email_value = ""
    if request.method == 'POST':
        email_value = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        # Проверяем, что оба поля заполнены
        if not email_value or not password:
            errors.append("Пожалуйста, заполните все поля.")
        else:
            user = get_user_by_email(db.session, email_value)
            if user and check_password_hash(user.password_hash, password):
                session['user_id'] = user.id
                return redirect(url_for('index.index'))
            else:
                errors.append("Неверная почта или пароль.")
    return render_template('login.html', errors=errors, email=email_value)


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    errors = []  # Список для хранения сообщений об ошибках
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Простейшая проверка: все поля заполнены
        if not username or not email or not password or not confirm_password:
            errors.append("Все поля обязательны для заполнения.")
        # Проверка совпадения паролей
        if password != confirm_password:
            errors.append("Пароль и его подтверждение не совпадают.")
        # Проверка существования пользователя
        if get_user_by_email(db.session, email):
            errors.append("Пользователь с такой почтой уже существует.")

        if errors:
            # Если есть ошибки, отрисовываем шаблон с переменной errors
            return render_template('register.html', errors=errors, username=username, email=email)

        # Если ошибок нет, регистрируем пользователя
        password_hash = generate_password_hash(password)
        create_user(db.session, username, email, password_hash)
        return redirect(url_for('auth.login'))
    return render_template('register.html', errors=errors)


@auth_bp.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Вы успешно вышли из системы.', 'success')
    return redirect(url_for('auth.login'))


@auth_bp.route('/users')
def users():
    # Страница для отображения списка пользователей
    if 'user_id' not in session:
        flash('Необходимо выполнить вход.', 'error')
        return redirect(url_for('auth.login'))
    all_users = get_all_users(db.session)
    return render_template('users.html', users=all_users)


@auth_bp.route('/profile')
def profile():
    if 'user_id' not in session:
        flash('Необходимо выполнить вход.', 'danger')
        return redirect(url_for('auth.login'))
    user = db.session.query(User).get(session['user_id'])
    return render_template('profile.html', user=user)
