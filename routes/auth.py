import random
from urllib.parse import urlparse, urljoin

import flask
from flask import Blueprint, request, render_template, redirect, url_for, flash, session, abort, current_app
from flask_login import login_user, logout_user, login_required, current_user
from flask_mail import Message
from werkzeug.security import generate_password_hash, check_password_hash
from db.user_repo import get_user_by_email, create_user, get_all_users
from db.models import db, User
from helpers.login import admin_required
from services.mail_manager import mail

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
            if user and check_password_hash(user.password_hash, password):
                # Генерируем 2FA-код (например, 6-значный)
                code = str(random.randint(100000, 999999))

                # Сохраняем в сессии временно
                session['2fa_user_id'] = user.id
                session['2fa_code'] = code

                # Отправляем email
                msg = Message("Код подтверждения",
                              sender="ваша_почта@gmail.com",
                              recipients=[user.email])
                msg.body = f"Ваш код: {code}"
                mail.send(msg)

                # Перенаправляем на страницу ввода кода
                return redirect(url_for('auth.two_factor'))
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


@auth_bp.route('/2fa', methods=['GET', 'POST'])
def two_factor():
    if request.method == 'POST':
        input_code = request.form.get('code', '')
        saved_code = session.get('2fa_code', None)
        user_id = session.get('2fa_user_id', None)
        if not saved_code or not user_id:
            flash("2FA сессиясы ескірді, қайта кіріп көріңіз.", "error")
            return redirect(url_for('auth.login'))

        if input_code == saved_code:
            # Код верный, логиним
            user = User.query.get(user_id)
            if user:
                login_user(user)
                # очищаем код из сессии
                session.pop('2fa_code', None)
                session.pop('2fa_user_id', None)
                # Делаем редирект на главную или куда нужно
                return redirect(url_for('index.index'))
            else:
                flash("Пайдаланушы табылмады.", "error")
                return redirect(url_for('auth.login'))
        else:
            flash("Код қате!", "error")
            return redirect(url_for('auth.two_factor'))

    return render_template('two_factor.html')


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


@auth_bp.route('/admin/users')
@admin_required
def admin_users():
    all_users = User.query.all()
    return render_template('admin_users.html', users=all_users)


@auth_bp.route('/admin/users/change_role', methods=['POST'])
@admin_required
def admin_change_role():
    user_id = request.form.get('user_id')
    new_role = request.form.get('new_role')

    user = User.query.get(user_id)
    if not user:
        flash("Пайдаланушы табылмады.", "error")
        return redirect(url_for('auth.admin_users'))

    user.role = new_role
    db.session.commit()
    flash(f"Роль изменена на {new_role} для {user.username}", "success")
    return redirect(url_for('auth.admin_users'))


@auth_bp.route('/admin/users/delete', methods=['POST'])
@admin_required
def admin_delete_user():
    user_id = request.form.get('user_id')
    user = User.query.get(user_id)
    if not user:
        flash("Пайдаланушы табылмады.", "error")
        return redirect(url_for('auth.admin_users'))
    if user.id == current_user.id:
        flash("Невозможно удалить самого себя.", "error")
        return redirect(url_for('auth.admin_users'))

    db.session.delete(user)
    db.session.commit()
    flash(f"Пайдаланушы {user.username} жойылды.", "success")
    return redirect(url_for('auth.admin_users'))

