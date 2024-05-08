from app import app, db

def clear_database():
    with app.app_context():
        # Delete all records from each table
        for table in reversed(db.metadata.sorted_tables):
            db.session.execute(table.delete())
        db.session.commit()

if __name__ == "__main__":
    clear_database()
    print("Database cleared successfully.")
