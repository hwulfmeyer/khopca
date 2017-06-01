import helper

def clusterData():
    pass
    # 1. load data
    # 2. create adjacency
    # 3. create value "array" for data
    # 4. apply all rules to data until nothing changes
    # 5. outpust cluster


def applyRulesToData(adjmatrix, dataArray):
    x = dataArray.shape[0]
    for i in range(0,x,1):
        dataArray[i] = rule1(i, adjmatrix, dataArray)
        dataArray[i] = rule2(i, adjmatrix, dataArray)
        dataArray[i] = rule3(i, adjmatrix, dataArray)
        dataArray[i] = rule4(i, adjmatrix, dataArray)


def rule1(nodeid, adjmatrix, dataArray):
    """RULE 1
    if max(W(N(n))) > w_n
2     w_n = max(W(N(n))) - 1"""


def rule2(nodeid, adjmatrix, dataArray):
    """RULE 2
    if max(W(N(n)) == MIN & w_n == MIN
2     w_n = MAX;"""

def rule3(nodeid, adjmatrix, dataArray):
    """RULE 3
    if max(W(N(n))) <= w_n && w_n != MAX
2     w_n = w_n - 1;"""

def rule4(nodeid, adjmatrix, dataArray):
    """RULE 4
    if max(W(N(n)) == MAX && w_n == MAX
2     w_n = apply criterion to select a node from set (max(W(N(n)),w_n);
3     w_n = w_n - 1;"""