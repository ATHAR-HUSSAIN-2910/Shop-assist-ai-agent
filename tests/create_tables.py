from src.database.connection import engine
from src.database.models import Base


print("Creating database tables...")

Base.metadata.create_all(engine)

print("Database tables created successfully!")