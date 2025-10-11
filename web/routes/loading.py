"""
Loading routes for the dimetuverdad web application.
Contains loading page endpoints.
"""

from flask import Blueprint, render_template

loading_bp = Blueprint('loading', __name__)

@loading_bp.route('/loading/<message>')
def loading_page(message):
    """Show a loading page with a custom message."""
    return render_template('loading.html', message=message.replace('_', ' '))

@loading_bp.route('/loading')
def loading_default():
    """Show a default loading page."""
    return render_template('loading.html', message="Cargando...")