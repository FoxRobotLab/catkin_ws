"""an emergency program for just in case you ran organizePics twice before mappingLocData. It writes 
a name change txt file that maps an images original name to its third name."""

prevChanges = open("nameChangesMay30-2.txt", "r")
updateChanges = open("nameChangesMay30.txt", "r")
combined = open("may30nameChanges.txt", "w")

nameDict = {}

for name in prevChanges.readlines():
    names = name.split()
    print names
    if len(names) is not 0:
        nameDict[names[0]] = names[1]


for name in updateChanges.readlines():
    names = name.split()
    if names[1] not in nameDict.keys():
        combined.write(str(int(names[0])) + " " + str(int(names[1])))
        combined.write('\n')
    else:
        combined.write(str(names[0]) + " " + str(nameDict[names[1]]))
        combined.write('\n')


prevChanges.close()
updateChanges.close()
combined.close()

