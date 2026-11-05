from datetime import date

from sqlalchemy import Date, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Order(Base):
    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True
    )

    customer_id: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True          # Speeds up "all orders for this customer" queries
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True          # Speeds up filtering by order status
    )

    tracking_number: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True
    )

    estimated_delivery: Mapped[date | None] = mapped_column(
        Date,
        nullable=True
    )