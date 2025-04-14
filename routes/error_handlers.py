import werkzeug.exceptions
from flask import render_template, Blueprint


def page_not_found(e):
    return render_template('404.html'), 404


def unauthorized(e):
    return render_template('401.html'), 401


def forbidden(e):
    return render_template('403.html'), 403


def register_error_handlers(app):
    app.register_error_handler(404, page_not_found)
    app.register_error_handler(401, unauthorized)
    app.register_error_handler(403, forbidden)
