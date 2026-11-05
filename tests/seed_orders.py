from datetime import date

from src.database.connection import SessionLocal
from src.database.models import Order


orders = [
    Order(
        order_id="1024",
        customer_id="customer_001",
        status="shipped",
        tracking_number="TRK1024",
        estimated_delivery=date(2026, 9, 20)
    ),
    Order(
        order_id="1025",
        customer_id="customer_002",
        status="processing",
        tracking_number=None,
        estimated_delivery=date(2026, 9, 22)
    ),
    Order(
        order_id="1026",
        customer_id="customer_003",
        status="delivered",
        tracking_number="TRK1026",
        estimated_delivery=date(2026, 9, 15)
    ),
    Order(
    order_id="1027",
    customer_id="customer_004",
    status="processing",
    tracking_number=None,
    estimated_delivery=date(2026, 9, 23)
    )
]


with SessionLocal() as session:

    session.add_all(orders)

    session.commit()

    print("Sample orders inserted successfully!")