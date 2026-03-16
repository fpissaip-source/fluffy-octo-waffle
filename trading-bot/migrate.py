#!/usr/bin/env python3
"""Database migration script for Trading Bot.

Creates all required tables if they don't exist.
Safe to run multiple times — uses CREATE TABLE IF NOT EXISTS via SQLAlchemy.

Usage:
    python3 migrate.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from database import init_db, Base, get_engine

def main():
    if not Config.DATABASE_URL:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    print("Running database migration...")
    print(f"Database: {Config.DATABASE_URL[:30]}...")

    engine = get_engine()
    Base.metadata.create_all(engine)

    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    bot_tables = [t for t in tables if t.startswith("bot_")]

    print(f"\nBot tables ({len(bot_tables)}):")
    for t in sorted(bot_tables):
        cols = inspector.get_columns(t)
        print(f"  {t} ({len(cols)} columns)")
        for col in cols:
            print(f"    - {col['name']}: {col['type']}")

    print("\nMigration complete.")


if __name__ == "__main__":
    main()
