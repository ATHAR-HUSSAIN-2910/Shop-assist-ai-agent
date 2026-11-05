from datetime import date, timedelta

from sqlalchemy import select

from src.database.connection import SessionLocal
from src.database.models import Order


START_ORDER_ID = 2001
END_ORDER_ID = 2300


def build_order(order_number: int) -> Order:
    index = order_number - START_ORDER_ID

    # A predictable mix of statuses makes the demo data useful for testing.
    status_cycle = ["processing", "shipped", "delivered", "cancelled"]
    status = status_cycle[index % len(status_cycle)]

    customer_number = (index % 100) + 1

    tracking_number = None
    if status in {"shipped", "delivered"}:
        tracking_number = f"TRK{order_number}"

    estimated_delivery = None
    if status == "processing":
        estimated_delivery = date.today() + timedelta(days=2 + (index % 5))
    elif status == "shipped":
        estimated_delivery = date.today() + timedelta(days=1 + (index % 4))
    elif status == "delivered":
        estimated_delivery = date.today() - timedelta(days=1 + (index % 10))

    return Order(
        order_id=str(order_number),
        customer_id=f"customer_{customer_number:03d}",
        status=status,
        tracking_number=tracking_number,
        estimated_delivery=estimated_delivery
    )


def seed_demo_orders():
    inserted = 0
    skipped = 0

    with SessionLocal() as session:
        existing_ids = set(
            session.scalars(
                select(Order.order_id).where(
                    Order.order_id.between(
                        str(START_ORDER_ID),
                        str(END_ORDER_ID)
                    )
                )
            ).all()
        )

        for order_number in range(START_ORDER_ID, END_ORDER_ID + 1):
            order_id = str(order_number)

            if order_id in existing_ids:
                skipped += 1
                continue

            session.add(build_order(order_number))
            inserted += 1

        session.commit()

    print(f"Demo order range: {START_ORDER_ID}-{END_ORDER_ID}")
    print(f"Inserted: {inserted}")
    print(f"Skipped existing: {skipped}")
    print(f"Total demo orders available in this range: {END_ORDER_ID - START_ORDER_ID + 1}")


if __name__ == "__main__":
    seed_demo_orders()
