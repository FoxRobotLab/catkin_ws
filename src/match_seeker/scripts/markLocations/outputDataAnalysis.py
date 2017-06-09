
countEvals = 0.0
countNormDiffs = 0
countGoodScoreDiffs = 0
goodScoreDiffs = []
fewPointsCount = 0
goodMatchList = []
allMatchList = []
alt1MatchList = []
alt2MatchList = []
matchScoreList = []
maxGoodMatch = 0

def processEvalOutput(lines, index):
    global countNormDiffs, countGoodScoreDiffs, fewPointsCount, goodMatchList, allMatchList
    global goodScoreDiffs, matchScoreList, countEvals, maxGoodMatch
    countEvals += 1

    myLabel, myDesCnt = getParts(lines[index])
    otherLabel, otherDesCnt = getParts(lines[index+1])
    goodLabel, goodMat = getParts(lines[index+2])
    alt1Label,alt1Mat = getParts(lines[index+3])
    alt2Label, alt2Mat = getParts(lines[index+4])
    allLabel, allMat = getParts(lines[index+5])
    scoreLabel, score = getParts(lines[index+6])
    # rawLabel, rawS = getParts(lines[index+5])
    # normLabel, normS = getParts(lines[index+6])

    if myDesCnt < 500 or otherDesCnt < 500:
        fewPointsCount += 1
    goodMatchList.append(goodMat)
    alt1MatchList.append(alt1Mat)
    alt2MatchList.append(alt2Mat)
    allMatchList.append(allMat)
    matchScoreList.append(score)
    maxGoodMatch = max(maxGoodMatch, goodMat)
    if goodMat >= 100:
        countGoodScoreDiffs += 1
    # if normS != rawS:
    #     print "Normed score differs from raw score: ", normS, rawS
    #     countNormDiffs += 1
    return index+7


def getParts(line):
    [label, numStr] = line.split(":")
    num = float(numStr.strip())
    return label, num


outFile = open("../../res/logs/Jun09Fri-141402.txt", 'r')
lines = outFile.readlines()
outFile.close()


i = 0
while i < len(lines):
    line = lines[i]
    if "evaluateSimilarity" in line:
        i = processEvalOutput(lines, i+1)
    else:
        i += 1

print "TotalEvaluations:", countEvals
print "Max matches:", maxGoodMatch
print "Percent over 100 good matches:", countGoodScoreDiffs / countEvals
print "Percent with few good matches:", fewPointsCount / countEvals
print "Percent where norming mattered:", countNormDiffs / countEvals


