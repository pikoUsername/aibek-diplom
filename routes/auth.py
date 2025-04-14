from urllib.parse import urlparse, urljoin

import flask
from flask import Blueprint, request, render_template, redirect, url_for, flash, session, abort, current_app
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from db.user_repo import get_user_by_email, create_user, get_all_users
from db.models import db, User

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    errors = []
    email_value = ""
    if request.method == 'POST':
        email_value = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not email_value or not password:
            errors.append("Барлық өрістерді толтырыңыз.")
        else:
            user = get_user_by_email(db.session, email_value)
            # Для отладки: проверьте, что user получен
            print("Получен пользователь:", user)
            if user and check_password_hash(user.password_hash, password):
                login_user(user, remember=True, force=True)
                next_url = request.args.get('next')
                if next_url and not is_safe_url(next_url):
                    return abort(400)
                return flask.redirect(next_url or flask.url_for('index.index'))
            else:
                errors.append("Қате пошта немесе құпия сөз.")
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
        user = create_user(db.session, username, email, password_hash)

        login_user(user, remember=True)

        return redirect(url_for('index.index'))
    return render_template('register.html', errors=errors)


@auth_bp.route('/logout')
def logout():
    logout_user()
    flash('Вы успешно вышли из системы.', 'success')
    return redirect(url_for('auth.login'))


@auth_bp.route('/profile')
@login_required
def profile():
    user = db.session.query(User).get(current_user.id)
    return render_template('profile.html', user=user)


@auth_bp.route('/users')
@login_required
def users():
    all_users = get_all_users(db.session)
    return render_template('users.html', users=all_users)


@auth_bp.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    errors = []
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        bio = request.form.get('bio', '').strip()
        social_link = request.form.get('social_link', '').strip()

        # Проверка обязательных полей
        if not username or not email:
            errors.append("Пайдаланушының аты мен Email өрістері толтырылуы қажет!")

        # Проверка уникальности username, если он изменён
        if username != current_user.username:
            user_with_username = User.query.filter(User.username == username).first()
            if user_with_username:
                errors.append("Бұл пайдаланушы аты бұрын қолданылған!")

        # Проверка уникальности email, если он изменён
        if email != current_user.email:
            user_with_email = User.query.filter(User.email == email).first()
            if user_with_email:
                errors.append("Бұл email бұрын қолданылған!")

        if not errors:
            try:
                current_user.username = username
                current_user.email = email
                current_user.bio = bio
                current_user.social_link = social_link
                db.session.commit()
                flash("Профиль сәтті жаңартылды!", "success")
                return redirect(url_for('auth.profile'))
            except Exception as e:
                db.session.rollback()
                current_app.logger.exception("Профильді жаңарту кезінде қате: %s", e)
                errors.append("Профильді жаңартқанда қате орын алды!")

    return render_template('edit_profile.html', user=current_user, errors=errors)