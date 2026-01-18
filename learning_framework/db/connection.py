"""PostgreSQL connection pool management"""

import os
from contextlib import contextmanager
from typing import Optional

# Optional psycopg2 import - falls back gracefully for local development
try:
    import psycopg2
    from psycopg2 import pool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None
    pool = None


# Global connection pool
_connection_pool: Optional['pool.ThreadedConnectionPool'] = None


def init_pool(min_conn: int = 1, max_conn: int = 10) -> bool:
    """Initialize the connection pool from DATABASE_URL environment variable

    Args:
        min_conn: Minimum connections to keep open
        max_conn: Maximum connections allowed

    Returns:
        True if pool initialized successfully, False otherwise
    """
    global _connection_pool

    if not PSYCOPG2_AVAILABLE:
        print("Warning: psycopg2 not available, database features disabled")
        return False

    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("Warning: DATABASE_URL not set, database features disabled")
        return False

    try:
        # Connection options for Railway (handle cold starts)
        _connection_pool = pool.ThreadedConnectionPool(
            min_conn,
            max_conn,
            database_url,
            connect_timeout=10,  # 10 second connection timeout
            options='-c statement_timeout=30000'  # 30 second query timeout
        )
        print(f"Database pool initialized (min={min_conn}, max={max_conn})")
        return True
    except Exception as e:
        print(f"Failed to initialize database pool: {e}")
        return False


def close_pool():
    """Close all connections in the pool"""
    global _connection_pool
    if _connection_pool:
        _connection_pool.closeall()
        _connection_pool = None
        print("Database pool closed")


def get_connection():
    """Get a connection from the pool

    Returns:
        A database connection, or None if pool not initialized
    """
    if _connection_pool is None:
        return None
    return _connection_pool.getconn()


def return_connection(conn):
    """Return a connection to the pool"""
    if _connection_pool and conn:
        _connection_pool.putconn(conn)


@contextmanager
def db_connection():
    """Context manager for database connections

    Usage:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ...")

    Yields:
        Database connection (auto-returned to pool on exit)
    """
    conn = get_connection()
    if conn is None:
        yield None
        return

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        return_connection(conn)


def is_pool_available() -> bool:
    """Check if database pool is available"""
    return _connection_pool is not None
