from typing import Optional

from sqlalchemy import select

from src.database.connection import SessionLocal
from src.database.models import Order


def track_order(order_id: str) -> Optional[dict]:
    """
    Get tracking information for an order from PostgreSQL.
    """

    with SessionLocal() as session:

        statement = select(Order).where(
            Order.order_id == order_id
        )

        order = session.scalar(statement)

        if order is None:
            return None

        return {
            "order_id": order.order_id,
            "status": order.status,
            "tracking_number": order.tracking_number,
            "estimated_delivery": (
                order.estimated_delivery.isoformat()
                if order.estimated_delivery
                else None
            )
        }


def cancel_order(order_id: str) -> dict:
    """
    Cancel an order if it is still processing.
    """

    with SessionLocal() as session:
        statement = select(Order).where(
            Order.order_id == order_id
        )

        order = session.scalar(statement)

        if order is None:
            return {
                "error": f"Order {order_id} was not found."
            }

        if order.status != "processing":
            return {
                "error": (
                    f"Order {order_id} cannot be cancelled "
                    f"because its current status is '{order.status}'."
                )
            }

        order.status = "cancelled"
        session.commit()

        return {
            "order_id": order.order_id,
            "status": order.status
        }