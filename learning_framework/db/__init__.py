"""Database module for PostgreSQL integration"""

from .connection import get_connection, init_pool, close_pool
from .models import ensure_schema, UserRepository, ProgressRepository, ReviewRepository

__all__ = [
    'get_connection',
    'init_pool',
    'close_pool',
    'ensure_schema',
    'UserRepository',
    'ProgressRepository',
    'ReviewRepository',
]
