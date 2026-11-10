"""2D face-orientation proxy, not gaze or measured Euler angles."""

from math import isfinite


def head_direction(points):
    if len(points) <= 263:
        return None
    a, b, nose, chin = points[33], points[263], points[1], points[152]
    if any(not isfinite(v) or not 0 <= v <= 1 for p in (a, b, nose, chin) for v in (p.x, p.y)):
        return None
    width = abs(b.x - a.x)
    height = chin.y - (a.y + b.y) / 2
    if width < 0.02 or height <= 0.02:
        return None
    horizontal = (nose.x - (a.x + b.x) / 2) / width
    vertical = (nose.y - (a.y + b.y) / 2) / height
    if horizontal > 0.2:
        return "right"
    if horizontal < -0.2:
        return "left"
    if vertical < 0.12:
        return "up"
    if vertical > 0.5:
        return "down"
    return "center"
