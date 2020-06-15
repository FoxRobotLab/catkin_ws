import math

def closest_bound_pt(robot_x, robot_y):

    cornerpts = [12, 8, 14, 10]
    # add/subtract 0.2 to nudge the robot back in bounds
    x1 = cornerpts[0] + 0.2
    x2 = cornerpts[2] - 0.2
    y1 = cornerpts[1] + 0.2
    y2 = cornerpts[3] - 0.2
    bounding_pts = [(x1, y1), (x1, y1 + .25 * (y2 - y1)), (x1, y1 + .5 * (y2 - y1)), (x1, y1 + .75 * (y2 - y1)),
                    (x1, y2), (x1 + .25 * (x2 - x1), y2), (x1 + .5 * (x2 - x1), y2), (x1 + .75 * (x2 - x1), y2),
                    (x2, y2), (x1 + .25 * (x2 - x1), y1), (x1 + .5 * (x2 - x1), y1), (x1 + .75 * (x2 - x1), y1),
                    (x2, y1), (x2, y1 + .25 * (y2 - y1)), (x2, y1 + .5 * (y2 - y1)), (x2, y1 + .75 * (y2 - y1))]
    xr = robot_x
    yr = robot_y
    best_dist = 99
    best_pt = None
    for pt in bounding_pts:
        x1 = pt[0]
        y1 = pt[1]
        dist = math.sqrt((x1 - xr) ** 2 + (y1 - yr) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_pt = (x1, y1)
    print ("Localizer 288 best pt: " + str(best_pt))
    return best_pt

if __name__ == '__main__':
    closest_bound_pt(15, 7)
    closest_bound_pt(16, 8)
    closest_bound_pt(17, 10)
    closest_bound_pt(9,9)
