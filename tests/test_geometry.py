from __future__ import annotations

from umafactor.core.geometry import Point, Rect, Size


def test_point_and_size_tuple_helpers() -> None:
    assert Point(2, 3).as_tuple() == (2, 3)
    assert Size(40, 20).as_tuple() == (40, 20)
    assert Size.from_image_shape((12, 34, 3)) == Size(width=34, height=12)


def test_rect_properties_and_tuple_conversion() -> None:
    rect = Rect.from_tuple((10, 20, 25, 44))

    assert rect.width == 15
    assert rect.height == 24
    assert rect.area == 360
    assert rect.is_empty is False
    assert rect.as_tuple() == (10, 20, 25, 44)


def test_rect_treats_inverted_dimensions_as_empty() -> None:
    rect = Rect(10, 20, 5, 20)

    assert rect.width == 0
    assert rect.height == 0
    assert rect.area == 0
    assert rect.is_empty is True


def test_rect_translate_scale_and_clamp() -> None:
    rect = Rect(-4, 8, 12, 30)

    assert rect.translate(dx=5, dy=-3) == Rect(1, 5, 17, 27)
    assert rect.scale(0.5) == Rect(-2, 4, 6, 15)
    assert rect.clamp(Size(width=10, height=20)) == Rect(0, 8, 10, 20)
