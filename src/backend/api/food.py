# File: backend/api/food.py
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

# Ta funkcja z 'database.py' będzie dostarczać sesję do bazy danych
from src.backend.infrastructure.database.database import get_db
from src.backend.schemas import shopping_schemas
from src.backend.services import shopping_service

router = APIRouter()


# Funkcja-zależność (dependency), która tworzy i zamyka sesję dla każdego zapytania


@router.post(
    "/shopping-trips/", response_model=shopping_schemas.ShoppingTrip, tags=["Food"]
)
async def add_shopping_trip(
    trip: shopping_schemas.ShoppingTripCreate, db: AsyncSession = Depends(get_db)
):
    """
    Dodaje nowy paragon z listą produktów do bazy danych.
    """
    return await shopping_service.create_shopping_trip(db=db, trip=trip)


@router.get(
    "/shopping-trips/",
    response_model=List[shopping_schemas.ShoppingTrip],
    tags=["Food"],
)
async def read_shopping_trips(
    skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)
):
    """
    Pobiera listę ostatnich paragonów z bazy danych.
    """
    trips = await shopping_service.get_shopping_trips(db=db, skip=skip, limit=limit)
    return trips


@router.patch(
    "/shopping-trips/{trip_id}",
    response_model=shopping_schemas.ShoppingTrip,
    tags=["Food"],
)
async def update_shopping_trip(
    trip_id: int,
    trip_update: shopping_schemas.ShoppingTripUpdate,
    db: AsyncSession = Depends(get_db),
):
    """
    Aktualizuje dane paragonu w bazie danych.
    Można zaktualizować tylko wybrane pola.
    """
    updated_trip = await shopping_service.update_shopping_trip(
        db=db, trip_id=trip_id, trip_update=trip_update
    )

    if updated_trip is None:
        raise HTTPException(
            status_code=404,
            detail="Paragon o podanym ID nie został znaleziony",
        )

    return updated_trip


@router.patch(
    "/products/{product_id}", response_model=shopping_schemas.Product, tags=["Food"]
)
async def update_product(
    product_id: int,
    product_update: shopping_schemas.ProductUpdate,
    db: AsyncSession = Depends(get_db),
):
    """
    Aktualizuje dane produktu w bazie danych.
    Można zaktualizować tylko wybrane pola.
    """
    updated_product = await shopping_service.update_product(
        db=db, product_id=product_id, product_update=product_update
    )

    if updated_product is None:
        raise HTTPException(
            status_code=404,
            detail="Produkt o podanym ID nie został znaleziony",
        )

    return updated_product


@router.delete("/shopping-trips/{trip_id}", status_code=200)
async def delete_shopping_trip(trip_id: int, db: AsyncSession = Depends(get_db)):
    """
    Endpoint do usunięcia pojedynczego paragonu.
    """
    success = await shopping_service.delete_shopping_trip(db=db, trip_id=trip_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Paragon o ID {trip_id} nie został znaleziony",
        )

    return {
        "status": "ok",
        "message": f"Paragon o ID {trip_id} został pomyślnie usunięty",
    }


@router.delete("/products/{product_id}", status_code=200)
async def delete_product(product_id: int, db: AsyncSession = Depends(get_db)):
    """
    Endpoint do usunięcia pojedynczego produktu.
    """
    success = await shopping_service.delete_product(db=db, product_id=product_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Produkt o ID {product_id} nie został znaleziony",
        )

    return {
        "status": "ok",
        "message": f"Produkt o ID {product_id} został pomyślnie usunięty",
    }
