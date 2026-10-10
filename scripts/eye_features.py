from math import dist


def calculate_ear(points):
    """Menghitung EAR dari enam titik 2D berurutan P1 sampai P6."""
    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = dist(p2, p6)
    vertical_2 = dist(p3, p5)
    horizontal = dist(p1, p4)

    if horizontal == 0.0:
        raise ValueError("Lebar mata harus lebih besar dari nol.")

    return (vertical_1 + vertical_2) / (2.0 * horizontal)
