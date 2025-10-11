"""
Utilities package for the dimetuverdad project.
"""

from .database import *
from .paths import *

# NOTE: We avoid importing .analyzer here to prevent circular imports when
# modules under the package (for example, `utils.text_utils`) are imported
# by higher-level modules like `analyzer`. Import `utils.analyzer`
# explicitly where needed instead (for example: `from utils import analyzer`).

__all__ = [
    # Database utilities
    'get_db_connection', 'get_tweet_data',

    # Path utilities
    'get_project_root', 'get_db_path',
]