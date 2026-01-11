from __future__ import annotations

M_PER_FT = 0.3048
M_PER_IN = 0.0254


def ft(value: float) -> float:
    return float(value) * M_PER_FT


def inch(value: float) -> float:
    return float(value) * M_PER_IN


def ft_in(feet: float, inches: float = 0.0) -> float:
    return ft(feet) + inch(inches)


def m_to_ft(value_m: float) -> float:
    return float(value_m) / M_PER_FT


def m_to_in(value_m: float) -> float:
    return float(value_m) / M_PER_IN

