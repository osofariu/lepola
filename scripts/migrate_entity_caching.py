#!/usr/bin/env python3
"""
Database migration script for entity caching feature.

This script adds the entities_source_analysis_id column to the analysis_results table
to support entity caching functionality.
"""

import sqlite3
import sys
from pathlib import Path

from src.core.config import settings


def migrate_database():
    """Add entities_source_analysis_id column to analysis_results table."""

    # Get database path
    database_url = settings.database_url
    if database_url.startswith("sqlite:///"):
        db_path = database_url[10:]  # Remove sqlite:///
    else:
        db_path = database_url

    print(f"Migrating database: {db_path}")

    # Check if database file exists
    if not Path(db_path).exists():
        print(f"Database file not found: {db_path}")
        print("Creating new database with updated schema...")
        return

    try:
        with sqlite3.connect(db_path) as db:
            # Check if column already exists
            cursor = db.execute("PRAGMA table_info(analysis_results)")
            columns = [row[1] for row in cursor.fetchall()]

            if "entities_source_analysis_id" in columns:
                print(
                    "✅ Column entities_source_analysis_id already exists. Migration not needed."
                )
                return

            print("Adding entities_source_analysis_id column...")

            # Add the new column
            db.execute(
                """
                ALTER TABLE analysis_results 
                ADD COLUMN entities_source_analysis_id TEXT
            """
            )

            # Add foreign key constraint
            db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_analysis_results_entities_source 
                ON analysis_results(entities_source_analysis_id)
            """
            )

            db.commit()
            print("✅ Successfully added entities_source_analysis_id column")

            # Verify the column was added
            cursor = db.execute("PRAGMA table_info(analysis_results)")
            columns = [row[1] for row in cursor.fetchall()]

            if "entities_source_analysis_id" in columns:
                print("✅ Migration completed successfully!")
            else:
                print("❌ Migration failed - column not found after addition")
                sys.exit(1)

    except sqlite3.Error as e:
        print(f"❌ Database migration failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error during migration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("Entity Caching Database Migration")
    print("=================================")
    print(
        "This script adds the entities_source_analysis_id column to support entity caching."
    )
    print()

    migrate_database()

    print("\nMigration completed!")
    print("You can now use the entity caching feature.")
