

def get_relative_pos(p1, p2) -> [int, int]:
    """
    returns relative position of two points
    """
    x1, y1 = p1
    x2, y2 = p2
    relative_x = x2 - x1
    relative_y = y2 - y1
    return (relative_x, relative_y)

def relative_moore_to_linear(p, radius):
    """
    compute index of a point in a moore neighborhood
    indexes the neighborhood on a row-by-row basis
    """
    x, y = p
    x_shifted = x + radius
    y_shifted = y + radius
    linear_index = y_shifted * (2 * radius + 1) + x_shifted
    return linear_index
