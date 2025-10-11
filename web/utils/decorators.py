"""
Decorators and error handlers for the Flask application.

This module contains reusable decorators for authentication, validation,
error handling, and rate limiting, as well as error handler functions.
"""

import time
import sqlite3
from functools import wraps
from typing import Callable, Any
from flask import flash, redirect, url_for, request, session, render_template, current_app

# In-memory store for rate limiting (in production, use Redis)
rate_limit_store = {}

# Analysis categories for validation
ANALYSIS_CATEGORIES = [
    'hate_speech', 'disinformation', 'conspiracy_theory',
    'far_right_bias', 'call_to_action', 'general'
]


def admin_required(f: Callable) -> Callable:
    """Decorator to require admin access for certain routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_authenticated'):
            flash('Acceso administrativo requerido', 'error')
            return redirect(url_for('admin.admin_login'))
        return f(*args, **kwargs)
    return decorated_function


def validate_input(*required_params: str) -> Callable:
    """Decorator to validate required parameters in request."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            for param in required_params:
                # Allow parameter to come from form, query args, or route/view args
                present = False
                if param in request.form:
                    present = True
                elif param in request.args:
                    present = True
                else:
                    # Flask route params are in request.view_args (e.g., /user/<username>)
                    view_args = getattr(request, 'view_args', {}) or {}
                    if param in view_args:
                        present = True

                if not present:
                    flash(f'Parámetro requerido faltante: {param}', 'error')
                    return redirect(request.referrer or url_for('main.index'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def handle_db_errors(f: Callable) -> Callable:
    """Decorator to handle database errors gracefully."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except sqlite3.OperationalError as e:
            current_app.logger.error(f"Database operational error: {str(e)}")
            if "locked" in str(e).lower():
                flash('La base de datos está ocupada. Por favor, inténtalo de nuevo en unos momentos.', 'warning')
            else:
                flash('Error de base de datos. Por favor, inténtalo de nuevo.', 'error')
            return redirect(request.referrer or url_for('main.index'))
        except sqlite3.IntegrityError as e:
            current_app.logger.error(f"Database integrity error: {str(e)}")
            flash('Error de integridad de datos. La operación no se pudo completar.', 'error')
            return redirect(request.referrer or url_for('main.index'))
        except Exception as e:
            current_app.logger.error(f"Unexpected database error: {str(e)}")
            flash('Ha ocurrido un error inesperado. Por favor, inténtalo de nuevo.', 'error')
            return redirect(request.referrer or url_for('main.index'))
    return decorated_function


def rate_limit(max_requests: int = 10, window_seconds: int = 60) -> Callable:
    """Rate limiting decorator for admin endpoints."""
    def decorator(f):
        @wraps(f)
        def wrapped_function(*args, **kwargs):
            # Get client identifier (IP address)
            client_id = request.remote_addr

            # Create key for this endpoint and client
            key = f"{client_id}:{request.endpoint}"

            # Get current time
            now = time.time()

            # Clean up old entries (simple in-memory store)
            if key not in rate_limit_store:
                rate_limit_store[key] = []

            # Remove entries older than window
            rate_limit_store[key] = [t for t in rate_limit_store[key] if now - t < window_seconds]

            # Check if rate limit exceeded
            if len(rate_limit_store[key]) >= max_requests:
                current_app.logger.warning(f"Rate limit exceeded for {client_id} on {request.endpoint}")
                return render_template('error.html',
                                     error_code=429,
                                     error_title="Demasiadas solicitudes",
                                     error_message="Has excedido el límite de solicitudes. Por favor, espera un momento antes de intentar de nuevo.",
                                     error_icon="fas fa-shield-alt",
                                     show_back_button=True), 429

            # Add current request timestamp
            rate_limit_store[key].append(now)

            return f(*args, **kwargs)
        return wrapped_function
    return decorator


# Error handler functions (to be registered with the Flask app)

def handle_database_operational_error(error: sqlite3.OperationalError) -> str:
    """Handle SQLite operational errors (database locked, corrupted, etc.)."""
    current_app.logger.error(f"Database operational error: {str(error)}")

    if "database is locked" in str(error).lower():
        return render_template('error.html',
                             error_code=503,
                             error_title="Base de datos ocupada",
                             error_message="La base de datos está siendo utilizada por otro proceso. Por favor, inténtalo de nuevo en unos momentos.",
                             error_icon="fas fa-database",
                             show_back_button=True), 503
    elif "no such table" in str(error).lower():
        return render_template('error.html',
                             error_code=500,
                             error_title="Error de esquema de base de datos",
                             error_message="La estructura de la base de datos no es correcta. Contacta al administrador.",
                             error_icon="fas fa-database",
                             show_back_button=True), 500
    else:
        return render_template('error.html',
                             error_code=500,
                             error_title="Error de base de datos",
                             error_message="Ha ocurrido un error en la base de datos. Por favor, inténtalo de nuevo más tarde.",
                             error_icon="fas fa-database",
                             show_back_button=True), 500


def handle_integrity_error(error: sqlite3.IntegrityError) -> str:
    """Handle SQLite integrity constraint violations."""
    current_app.logger.error(f"Database integrity error: {str(error)}")

    return render_template('error.html',
                         error_code=400,
                         error_title="Error de integridad de datos",
                         error_message="Los datos proporcionados violan las restricciones de integridad. Verifica la información e inténtalo de nuevo.",
                         error_icon="fas fa-exclamation-circle",
                         show_back_button=True), 400


def handle_timeout_error(error: TimeoutError) -> str:
    """Handle timeout errors for long-running operations."""
    current_app.logger.error(f"Timeout error: {str(error)}")

    return render_template('error.html',
                         error_code=504,
                         error_title="Tiempo de espera agotado",
                         error_message="La operación está tardando demasiado tiempo. Por favor, inténtalo de nuevo más tarde o contacta al administrador.",
                         error_icon="fas fa-clock",
                         show_back_button=True), 504


def handle_memory_error(error: MemoryError) -> str:
    """Handle memory exhaustion errors."""
    current_app.logger.error(f"Memory error: {str(error)}")

    return render_template('error.html',
                         error_code=507,
                         error_title="Error de memoria insuficiente",
                         error_message="El servidor no tiene suficiente memoria para procesar la solicitud. Por favor, inténtalo de nuevo más tarde.",
                         error_icon="fas fa-memory",
                         show_back_button=True), 507


def handle_forbidden_error(error) -> str:
    """Handle 403 errors."""
    return render_template('error.html',
                         error_code=403,
                         error_title="Acceso denegado",
                         error_message="No tienes permisos para acceder a esta página."), 403


def register_error_handlers(app) -> None:
    """Register all error handlers with the Flask application."""
    app.errorhandler(sqlite3.OperationalError)(handle_database_operational_error)
    app.errorhandler(sqlite3.IntegrityError)(handle_integrity_error)
    app.errorhandler(TimeoutError)(handle_timeout_error)
    app.errorhandler(MemoryError)(handle_memory_error)
    app.errorhandler(403)(handle_forbidden_error)