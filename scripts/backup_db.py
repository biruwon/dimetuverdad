#!/usr/bin/env python3
"""
Database backup script for dimetuverdad.
Creates timestamped backups of the development database.
"""

import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

# Import utility modules
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils import paths

def create_backup():
    """Create a timestamped backup of the database."""
    db_path = paths.get_db_path()
    backup_dir = paths.get_backup_dir()

    # Ensure backup directory exists
    os.makedirs(backup_dir, exist_ok=True)

    # Check if database exists
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return False

    # Create timestamped backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_filename = os.path.basename(db_path)
    backup_filename = f"{os.path.splitext(db_filename)[0]}_{timestamp}.db"
    backup_path = os.path.join(backup_dir, backup_filename)

    try:
        # Copy database file
        shutil.copy2(db_path, backup_path)

        # Get file size
        file_size = os.path.getsize(backup_path)
        file_size_mb = file_size / (1024 * 1024)

        print("✅ Database backup created successfully!")
        print(f"   📁 Location: {backup_path}")
        print(".2f")
        print(f"   📅 Timestamp: {timestamp}")

        return True

    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return False

def list_backups():
    """List existing backups."""
    backup_dir = paths.get_backup_dir()

    if not os.path.exists(backup_dir):
        print(f"📁 No backup directory found: {backup_dir}")
        return

    backups = [f for f in os.listdir(backup_dir) if f.endswith('.db')]
    backups.sort(reverse=True)  # Most recent first

    if not backups:
        print("📁 No backups found")
        return

    print(f"📁 Found {len(backups)} backups in {backup_dir}:")
    for backup in backups[:10]:  # Show last 10
        backup_path = os.path.join(backup_dir, backup)
        file_size = os.path.getsize(backup_path)
        file_size_mb = file_size / (1024 * 1024)
        print(".2f")
def cleanup_old_backups(keep_count=10):
    """Remove old backups, keeping only the most recent ones."""
    backup_dir = paths.get_backup_dir()

    if not os.path.exists(backup_dir):
        return

    backups = [f for f in os.listdir(backup_dir) if f.endswith('.db')]
    backups.sort(reverse=True)  # Most recent first

    if len(backups) <= keep_count:
        return

    to_remove = backups[keep_count:]
    print(f"🗑️  Removing {len(to_remove)} old backups...")

    for backup in to_remove:
        backup_path = os.path.join(backup_dir, backup)
        try:
            os.remove(backup_path)
            print(f"   ✅ Removed: {backup}")
        except Exception as e:
            print(f"   ❌ Failed to remove {backup}: {e}")

def show_usage():
    """Show usage information."""
    print("""
🛡️  dimetuverdad Database Backup Tool

Creates timestamped backups of the development database.

Usage:
    python backup_db.py [command]

Commands:
    (no command)    Create a new backup
    list           List existing backups
    cleanup        Remove old backups (keep last 10)

Examples:
    python backup_db.py              # Create backup
    python backup_db.py list         # List backups
    python backup_db.py cleanup      # Clean old backups

Backup location: ./backups/
Gitignored: Yes (backups not committed to repo)
""")

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else None

    if command == "list":
        list_backups()
    elif command == "cleanup":
        cleanup_old_backups()
    elif command is None:
        success = create_backup()
        if success:
            cleanup_old_backups()  # Clean up after successful backup
        sys.exit(0 if success else 1)
    else:
        show_usage()
        sys.exit(1)