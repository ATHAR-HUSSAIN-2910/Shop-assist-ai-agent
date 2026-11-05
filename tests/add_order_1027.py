from datetime import date

from src.database.connection import SessionLocal
from src.database.models import Order


order = Order(
    order_id="1027",
    customer_id="customer_004",
    status="processing",
    tracking_number=None,
    estimated_delivery=date(2026, 9, 23)
)

with SessionLocal() as session:
    session.add(order)
    session.commit()

    print("Order 1027 inserted successfully!")