######################################
# Graph of all of campus, with 102 nodes, so far!
#
"""Contains a graph of 102 points from the main part of the
Macalester campus.  The positions given are the latitude and
longitude values for each location, just the seconds part.  We can
assume, given the small scale of things, that these are fixed, flat
offsets from some (0,0), which would be southeast of campus,
on the east side of Snelling across from Berkeley St.  This is
where 44 degrees 56 minutes and 0 seconds North by 93 degrees
10 minutes and 0 seconds West is located."""

import math
from MapGraph import *

# The node data is a list of tuples that give latitude and longitude
nodeLocs = [(10.76, 11.85),  # Node 0
            (11.29, 11.74),  # Node 1
            (11.29, 11.20),  # Node 2
            (11.29, 11.02),  # Node 3
            (10.90, 8.55),   # Node 4
	    (10.76, 8.56),   # Node 5
	    (10.90, 7.83),   # Node 6
	    (10.76, 7.83),   # Node 7
	    (10.76, 5.47),   # Node 8
	    (11.72, 5.49),   # Node 9
	    (11.72, 5.59),   # Node 10
	    (12.62, 5.52),   # Node 11
	    (13.33, 11.14),   # Node 12
	    (11.63, 12.49),   # Node 13
	    (11.66, 11.18),   # Node 14
	    (11.65, 13.19),   # Node 15
	    (14.71, 13.78),   # Node 16
	    (14.71, 11.94),   # Node 17
	    (13.33, 9.24),   # Node 18
	    (13.37, 8.21),   # Node 19
	    (13.37, 7.89),   # Node 20
	    (12.82, 7.83),   # Node 21
	    (12.82, 8.20),   # Node 22
	    (12.65, 8.21),   # Node 23
	    (12.82, 9.07),   # Node 24
	    (12.23, 9.06),   # Node 25
	    (12.22, 8.38),   # Node 26
	    (10.82, 1.86),   # Node 27
	    (13.75, 1.87),   # Node 28
	    (13.75, 1.99),   # Node 29
	    (17.19, 1.86),   # Node 30
	    (12.59, 4.64),   # Node 31
	    (14.11, 4.57),   # Node 32
	    (14.38, 3.80),   # Node 33
	    (14.40, 3.61),   # Node 34
	    (16.01, 4.01),   # Node 35
	    (17.22, 4.01),   # Node 36
	    (17.41, 4.88),   # Node 37
	    (15.20, 7.96),   # Node 38
	    (15.19, 9.26),   # Node 39
	    (15.19, 9.86),   # Node 40
	    (15.82, 7.91),   # Node 41
	    (16.72, 8.14),   # Node 42
	    (16.98, 8.14),   # Node 43
	    (16.99, 8.92),   # Node 44
	    (16.98, 11.26),   # Node 45
	    (15.64, 11.24),   # Node 46
	    (16.99, 13.51),   # Node 47
	    (17.42, 6.44),   # Node 48
	    (17.93, 6.33),   # Node 49
	    (17.80, 7.03),   # Node 50
	    (17.95, 7.17),   # Node 51
	    (17.84, 8.32),   # Node 52
	    (17.83, 8.92),   # Node 53
	    (18.17, 6.20),   # Node 54
	    (18.39, 6.25),   # Node 55
	    (18.42, 6.56),   # Node 56
	    (18.27, 7.15),   # Node 57
	    (18.28, 8.34),   # Node 58
	    (18.78, 8.34),   # Node 59
	    (18.79, 8.14),   # Node 60
	    (19.42, 8.36),   # Node 61
	    (19.45, 7.17),   # Node 62
	    (19.34, 7.19),   # Node 63
	    (19.45, 6.27),   # Node 64
	    (18.77, 6.07),   # Node 65
	    (20.01, 4.32),   # Node 66
	    (20.00, 4.01),   # Node 67
	    (20.12, 4.00),   # Node 68
	    (18.63, 3.96),   # Node 69
	    (18.62, 2.99),   # Node 70
	    (20.14, 3.47),   # Node 71
	    (20.11, 1.87),   # Node 72
	    (19.41, 8.54),   # Node 73
	    (18.40, 9.11),   # Node 74
	    (18.42, 10.03),   # Node 75
	    (19.42, 10.03),   # Node 76
	    (19.42, 9.68),   # Node 77
	    (20.25, 10.01),   # Node 78
	    (20.04, 8.55),   # Node 79
	    (20.73, 8.49),   # Node 80
	    (20.72, 9.05),   # Node 81
	    (21.14, 8.41),   # Node 82
	    (21.98, 8.49),   # Node 83
	    (20.73, 10.04),   # Node 84
	    (22.01, 10.05),   # Node 85
	    (22.01, 9.87),   # Node 86
	    (23.87, 9.95),   # Node 87
	    (23.95, 8.47),   # Node 88
	    (22.91, 8.50),   # Node 89
	    (21.99, 7.15),   # Node 90
	    (22.32, 7.08),   # Node 91
	    (23.92, 6.21),   # Node 92
	    (23.93, 4.26),   # Node 93
	    (22.99, 4.28),   # Node 94
	    (23.93, 2.66),   # Node 95
	    (22.71, 4.30),   # Node 96
	    (22.49, 4.70),   # Node 97
	    (22.00, 4.89),   # Node 98
	    (21.38, 4.54),   # Node 99
	    (21.28, 4.29),   # Node 100
	    (20.78, 4.31),   # Node 101
	    (23.92, 1.86),   # Node 102
	    (11.66, 11.76),  # Node 103
            (21.99, 4.25)]   # Node 104



# Create the graph with all node data intact
macGraph = MapGraph(len(nodeLocs), nodeLocs)

# -----------------------
# Southwest Olin-Rice

# 0: 13, 5
macGraph.addEdge(0, 13)
macGraph.addEdge(0, 5)

# 1: 2, 103
macGraph.addEdge(1, 2)
macGraph.addEdge(1, 103)

# 2: 1, 3, 14
macGraph.addEdge(2, 3)
macGraph.addEdge(2, 14)

# 3: 2

# -----------------------
# South side:  4-8

# 4: 5
macGraph.addEdge(4, 5)

# 5: 0, 4, 7
macGraph.addEdge(5, 7)

# 6: 7
macGraph.addEdge(6, 7)

# 7: 5, 6, 8
macGraph.addEdge(7, 8)

# 8: 7, 9, 27
macGraph.addEdge(8, 9)
macGraph.addEdge(8, 27)

# -----------------------
# East side:  9-11

# 9: 8, 10, 11
macGraph.addEdge(9, 10)
macGraph.addEdge(9, 11)

# 10: 9

# 11: 9, 21, 31
macGraph.addEdge(11, 21)
macGraph.addEdge(11, 31)

# -----------------------
# Fine Arts/Janet Wallace:  12-17


# 12: 14, 18
macGraph.addEdge(12, 14)
macGraph.addEdge(12, 18)

# 13: 0, 15, 103
macGraph.addEdge(13, 15)
macGraph.addEdge(13, 103)

# 14: 2, 12, 103
macGraph.addEdge(14, 103)

# 103: 1, 13, 14

# 15: 13, 16
macGraph.addEdge(15, 16)

# 16: 15, 17
macGraph.addEdge(16, 17)

# 17: 16

# -----------------------
# North side:  18-22

# 18: 12, 19
macGraph.addEdge(18, 19)


# 19: 18, 20, 21
macGraph.addEdge(19, 20)
macGraph.addEdge(19, 21)

# 20: 19, 21, 38
macGraph.addEdge(20, 21)
macGraph.addEdge(20, 38)

# 21: 11, 20, 22, 23 
macGraph.addEdge(21, 22)
macGraph.addEdge(21, 23)

# 22: 19, 21, 23, 24
macGraph.addEdge(22, 23)
macGraph.addEdge(22, 24)

# 23:  21, 22

# -----------------------
# Rock garden:  24-26

# 24: 22, 25
macGraph.addEdge(24, 25)

# 25: 24, 26
macGraph.addEdge(25, 26)

# 26: 25

# -----------------------
# Gym nodes: 27-37

# 27: 8, 28
macGraph.addEdge(27, 28)

# 28: 27, 29, 30
macGraph.addEdge(28, 29)
macGraph.addEdge(28, 30)

# 29: 28

# 30: 28, 36, 72
macGraph.addEdge(30, 36)
macGraph.addEdge(30, 72)

# 31: 11, 32
macGraph.addEdge(31, 32)

# 32: 31, 33
macGraph.addEdge(32, 33)

# 33: 32, 34, 35
macGraph.addEdge(33, 34)
macGraph.addEdge(33, 35)

# 34: 33

# 35: 33, 36, 48
macGraph.addEdge(35, 36)
macGraph.addEdge(35, 48)

# 36: 30, 35, 37, 69
macGraph.addEdge(36, 37)
macGraph.addEdge(36, 69)


# 37: 36, 48
macGraph.addEdge(37, 48)


# -----------------------
# Fine arts nodes: 38-48

# 38: 20, 39, 41
macGraph.addEdge(38, 39)
macGraph.addEdge(38, 41)

# 39: 38, 40
macGraph.addEdge(39, 40)

# 40: 39

# 41: 38, 42, 48
macGraph.addEdge(41, 42)
macGraph.addEdge(41, 48)

# 42: 41, 43, 44
macGraph.addEdge(42, 43)
macGraph.addEdge(42, 44)

# 43: 42, 44, 50, 52
macGraph.addEdge(43, 44)
macGraph.addEdge(43, 50)
macGraph.addEdge(43, 52)

# 44: 42, 43, 45, 53
macGraph.addEdge(44, 45)
macGraph.addEdge(44, 53)

# 45: 44, 46, 47
macGraph.addEdge(45, 46)
macGraph.addEdge(45, 47)

# 46: 45

# 47: 45

# -----------------------
# North Shaw Field: 48-57

# 48: 35, 37, 41, 49, 50
macGraph.addEdge(48, 49)
macGraph.addEdge(48, 50)

# 49: 48, 50, 51, 54
macGraph.addEdge(49, 50)
macGraph.addEdge(49, 51)
macGraph.addEdge(49, 54)

# 50: 43, 48, 49, 51
macGraph.addEdge(50, 51)


# 51: 49, 50, 52, 56, 57
macGraph.addEdge(51, 52)
macGraph.addEdge(51, 56)
macGraph.addEdge(51, 57)

# 52: 43, 51, 53, 58
macGraph.addEdge(52, 53)
macGraph.addEdge(52, 58)

# 53: 44, 52, 58, 74
macGraph.addEdge(53, 58)
macGraph.addEdge(53, 74)

# 54: 49, 55
macGraph.addEdge(54, 55)

# 55: 54, 56, 65
macGraph.addEdge(55, 56)
macGraph.addEdge(55, 65)

# 56: 51, 55, 57
macGraph.addEdge(56, 57)

# 57: 51, 56, 58
macGraph.addEdge(57, 58)


# -----------------------
# Old Main/Library: 58-65


# 58: 52, 53, 57, 59, 74
macGraph.addEdge(58, 59)
macGraph.addEdge(58, 74)

# 59: 58, 60, 61, 74
macGraph.addEdge(59, 60)
macGraph.addEdge(59, 61)
macGraph.addEdge(59, 74)

# 60: 59

# 61: 59, 62, 73, 79
macGraph.addEdge(61, 62)
macGraph.addEdge(61, 73)
macGraph.addEdge(61, 79)

# 62: 61, 63, 64
macGraph.addEdge(62, 63)
macGraph.addEdge(62, 64)

# 63: 62

# 64: 62, 65, 66, 99
macGraph.addEdge(64, 65)
macGraph.addEdge(64, 66)
macGraph.addEdge(64, 99)

# 65: 55, 64

# -----------------------
# Kirk area:  66-72

# 66: 64, 67, 101
macGraph.addEdge(66, 67)
macGraph.addEdge(66, 101)


# 67: 66, 68, 69
macGraph.addEdge(67, 68)
macGraph.addEdge(67, 69)

# 68: 67, 71, 101
macGraph.addEdge(68, 71)
macGraph.addEdge(68, 101)

# 69: 36, 67, 70
macGraph.addEdge(69, 70)

# 70: 69

# 71: 68, 72, 101
macGraph.addEdge(71, 72)
macGraph.addEdge(71, 101)

# 72: 30, 71, 102
macGraph.addEdge(72, 102)


# -----------------------
# Carnegie:  73-81

# 73: 61

# 74: 53, 58, 59, 75
macGraph.addEdge(74, 75)

# 75: 74, 76
macGraph.addEdge(75, 76)

# 76: 75, 77, 78
macGraph.addEdge(76, 77)
macGraph.addEdge(76, 78)

# 77: 76

# 78: 76, 79, 84
macGraph.addEdge(78, 79)
macGraph.addEdge(78, 84)

# 79: 61, 78, 80, 81
macGraph.addEdge(79, 80)
macGraph.addEdge(79, 81)

# 80: 79, 81, 82
macGraph.addEdge(80, 81)
macGraph.addEdge(80, 82)

# 81: 79, 80, 84
macGraph.addEdge(81, 84)

# -----------------------
# Weyerhaeuser: 82-89

# 82: 62, 80, 83 
macGraph.addEdge(82, 83)

# 83: 82, 89, 90
macGraph.addEdge(83, 89)
macGraph.addEdge(83, 90)

# 84: 78, 81, 85
macGraph.addEdge(84, 85)

# 85: 84, 86, 87
macGraph.addEdge(85, 86)
macGraph.addEdge(85, 87)

# 86: 85

# 87: 85, 88, 89
macGraph.addEdge(87, 88)
macGraph.addEdge(87, 89)

# 88: 87, 89, 92
macGraph.addEdge(88, 89)
macGraph.addEdge(88, 92)

# 89: 83, 87, 88

# -----------------------
# Campus Center:  90-102


# 90: 83, 91, 98
macGraph.addEdge(90, 91)
macGraph.addEdge(90, 98)

# 91: 90

# 92: 88, 93, 97
macGraph.addEdge(92, 93)
macGraph.addEdge(92, 97)

# 93: 92, 94, 95
macGraph.addEdge(93, 94)
macGraph.addEdge(93, 95)

# 94: 93, 95, 96
macGraph.addEdge(94, 95)
macGraph.addEdge(94, 96)

# 95: 93, 94, 102
macGraph.addEdge(95, 102)

# 96: 94, 97, 104
macGraph.addEdge(96, 97)
macGraph.addEdge(96, 104)

# 97: 92, 96, 98, 104
macGraph.addEdge(97, 98)
macGraph.addEdge(97, 104)

# 98: 90, 97, 99, 104
macGraph.addEdge(98, 99)
macGraph.addEdge(98, 104)

# 99: 64, 98, 100, 104
macGraph.addEdge(99, 100)
macGraph.addEdge(99, 104)

# 100: 99, 101, 104
macGraph.addEdge(100, 101)
macGraph.addEdge(100, 104)

# 101: 66, 68, 71, 100

# 102:  72, 95

# 103: (See above)

# 104: 96, 97, 98, 99, 100



