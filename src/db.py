"""
SQLite storage for Sporty Hero crash game data.

Stores crash multiplier values scraped from the game,
along with round metadata for pattern analysis.
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent.parent / "data" / "sportyhero.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Get a database connection with WAL mode enabled."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS crashes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            round_id        TEXT UNIQUE,
            crash_value     REAL NOT NULL,
            timestamp       TEXT NOT NULL,
            scraped_at      TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS bets (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            round_id        TEXT,
            timestamp       TEXT NOT NULL,
            stake           REAL NOT NULL,
            cashout_target  REAL,
            crash_value     REAL,
            cashed_out      INTEGER DEFAULT 0,
            payout          REAL DEFAULT 0,
            profit          REAL DEFAULT 0,
            FOREIGN KEY (round_id) REFERENCES crashes(round_id)
        );

        CREATE INDEX IF NOT EXISTS idx_crashes_value ON crashes(crash_value);
        CREATE INDEX IF NOT EXISTS idx_crashes_timestamp ON crashes(timestamp);
    """)
    conn.commit()


def insert_crash(conn: sqlite3.Connection, round_id: str, crash_value: float,
                 timestamp: str = None) -> bool:
    """Insert a crash result. Returns False if already exists."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    scraped_at = datetime.now(timezone.utc).isoformat()
    try:
        conn.execute(
            "INSERT INTO crashes (round_id, crash_value, timestamp, scraped_at) VALUES (?, ?, ?, ?)",
            (round_id, crash_value, timestamp, scraped_at),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def insert_crashes_bulk(conn: sqlite3.Connection, crashes: list[dict]) -> int:
    """Insert multiple crash results. Returns count inserted."""
    count = 0
    for c in crashes:
        try:
            conn.execute(
                "INSERT INTO crashes (round_id, crash_value, timestamp, scraped_at) VALUES (?, ?, ?, ?)",
                (c.get("round_id", f"R{count}"),
                 c["crash_value"],
                 c.get("timestamp", datetime.now(timezone.utc).isoformat()),
                 datetime.now(timezone.utc).isoformat()),
            )
            count += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    return count


def get_total_stats(conn: sqlite3.Connection) -> dict:
    """Get overall summary statistics."""
    stats = {}
    row = conn.execute("SELECT COUNT(*) as n FROM crashes").fetchone()
    stats["total_rounds"] = row["n"]

    row = conn.execute("SELECT AVG(crash_value) as avg FROM crashes").fetchone()
    stats["avg_crash"] = row["avg"] or 0

    row = conn.execute("SELECT MIN(crash_value) as min FROM crashes").fetchone()
    stats["min_crash"] = row["min"] or 0

    row = conn.execute("SELECT MAX(crash_value) as max FROM crashes").fetchone()
    stats["max_crash"] = row["max"] or 0

    # Percentage of crashes below common thresholds
    for threshold in [1.5, 2.0, 3.0, 5.0, 10.0]:
        row = conn.execute(
            "SELECT COUNT(*) as n FROM crashes WHERE crash_value < ?",
            (threshold,)
        ).fetchone()
        total = stats["total_rounds"]
        stats[f"below_{threshold}x"] = row["n"] / total * 100 if total > 0 else 0

    return stats


def get_recent_crashes(conn: sqlite3.Connection, limit: int = 100) -> list[float]:
    """Get the most recent crash values."""
    rows = conn.execute(
        "SELECT crash_value FROM crashes ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    return [r["crash_value"] for r in reversed(rows)]


def get_all_crashes(conn: sqlite3.Connection) -> list[float]:
    """Get all crash values in order."""
    rows = conn.execute(
        "SELECT crash_value FROM crashes ORDER BY id"
    ).fetchall()
    return [r["crash_value"] for r in rows]
