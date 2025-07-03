from collections import deque


def find_all_reachable_neighbors(atom, getNeighbor):
    """
    Find all reachable atoms in a molecule, starting from atom
    :param atom: The starting atom
    :param getNeighbor: Function that returns all the neighbor of a node in a list
    :return: The list of all reachable atoms in the order of increasing distance
    """
    visitedSet = set([atom.GetIdx()])
    queue = deque([atom])
    returnList = []
    while queue:
        current_atom = queue.popleft()
        returnList.append(current_atom)
        neighbors = getNeighbor(current_atom)
        for neighbor in neighbors:
            if neighbor.GetIdx() not in visitedSet:
                visitedSet.add(neighbor.GetIdx())
                queue.append(neighbor)
    return returnList
