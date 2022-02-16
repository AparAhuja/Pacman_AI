


# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random
import util
import sys

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # check if it is a winning state, return 100 (reward) if True
        if successorGameState.isWin():
            return 100
        # determine ghost position
        ghostList = [ghostState.getPosition() for ghostState in newGhostStates]
        ghostDistance = [manhattanDistance(
            newPos, ghost) for ghost in ghostList]
        # ghost can potentially kill in this state, return -10 (penalty)
        if min(ghostDistance) < 2:
            return -10
        # not a winning state, hence, foodList is non-empty
        # check if newPos has food, return 10 if True (reward)
        if currentGameState.hasFood(newPos[0], newPos[1]):
            return 10
        # no win, no food, no ghost, give reward based on manhattan distance to nearest food
        foodList = newFood.asList()
        foodDistance = [manhattanDistance(newPos, food) for food in foodList]
        # more distance, less reward; less distance, more reward
        return 5 / min(foodDistance)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # check if game is over, or depth is 0
        if self.depth == 0 or gameState.isWin() or gameState.isLose():
            return None
        # store the number of agents (pacman and ghosts)
        numAgents = gameState.getNumAgents()
        legalMoves = gameState.getLegalActions(0)
        # create next agent
        nextAgent = MinimaxAgent()
        nextAgent.evaluationFunction = self.evaluationFunction
        # check for ghosts
        if numAgents > 1:
            nextAgent.index = 1
            nextAgent.depth = self.depth
        else:
            nextAgent.depth = self.depth - 1
        maxUtility = 0
        initialised = False
        for action in legalMoves:
            successorGameState = gameState.generateSuccessor(
                self.index, action)
            if initialised:
                temp = nextAgent.getUtility(successorGameState, numAgents)
                if temp > maxUtility:
                    maxUtility = temp
                    chosenAction = action
            else:
                maxUtility = nextAgent.getUtility(
                    successorGameState, numAgents)
                chosenAction = action
                initialised = True
        # return action resulting in maximum value (MAX)
        # print(maxUtility)
        return chosenAction

    def getUtility(self, gameState, numAgents=1):
        # check for leaf node
        if self.depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # get all legal moves (actions) for agent
        legalMoves = gameState.getLegalActions(self.index)
        # check for MAX agent (pacman)
        if self.index == 0:
            maxUtility = 0
            initialised = False
            # create next agent
            nextAgent = MinimaxAgent()
            nextAgent.evaluationFunction = self.evaluationFunction
            # check for ghosts
            if numAgents > 1:
                nextAgent.index = 1
                nextAgent.depth = self.depth
            else:
                # no ghosts, only pacman
                nextAgent.depth = self.depth - 1
            # determine max utility possible
            for action in legalMoves:
                successorGameState = gameState.generateSuccessor(
                    self.index, action)
                if initialised:
                    maxUtility = max(maxUtility, nextAgent.getUtility(
                        successorGameState, numAgents))
                else:
                    maxUtility = nextAgent.getUtility(
                        successorGameState, numAgents)
                    initialised = True
            # return utility value (MAX)
            return maxUtility
        else:
            # ghost (MIN) agent
            minUtility = 0
            initialised = False
            # create next agent
            nextAgent = MinimaxAgent()
            nextAgent.evaluationFunction = self.evaluationFunction
            # check for pacman (last ghost)
            if self.index + 1 == numAgents:
                nextAgent.depth = self.depth - 1
            else:
                # next agent is also ghost
                nextAgent.index = self.index + 1
                nextAgent.depth = self.depth
            # determine min utility possible
            for action in legalMoves:
                successorGameState = gameState.generateSuccessor(
                    self.index, action)
                if initialised:
                    minUtility = min(minUtility, nextAgent.getUtility(
                        successorGameState, numAgents))
                else:
                    minUtility = nextAgent.getUtility(
                        successorGameState, numAgents)
                    initialised = True
            # return utility value (MIN)
            return minUtility


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # check if game is over, or depth is 0
        if self.depth == 0 or gameState.isWin() or gameState.isLose():
            return None
        # store the number of agents (pacman and ghosts)
        numAgents = gameState.getNumAgents()
        legalMoves = gameState.getLegalActions(0)
        # create next agent
        nextAgent = AlphaBetaAgent()
        nextAgent.evaluationFunction = self.evaluationFunction
        # check for ghosts
        if numAgents > 1:
            nextAgent.index = 1
            nextAgent.depth = self.depth
        else:
            nextAgent.depth = self.depth - 1
        pathMaxUtility = -float('inf')
        pathMinUtility = float('inf')
        maxUtility = 0
        initialised = False
        for action in legalMoves:
            successorGameState = gameState.generateSuccessor(
                self.index, action)
            if initialised:
                temp = nextAgent.getUtility(
                    successorGameState, numAgents, pathMaxUtility, pathMinUtility)
                if temp > maxUtility:
                    maxUtility = temp
                    chosenAction = action
                # update pathMaxUtility appropriately
                pathMaxUtility = max(pathMaxUtility, maxUtility)
            else:
                maxUtility = nextAgent.getUtility(
                    successorGameState, numAgents, pathMaxUtility, pathMinUtility)
                chosenAction = action
                # update pathMaxUtility appropriately
                pathMaxUtility = max(pathMaxUtility, maxUtility)
                initialised = True
        # return action resulting in maximum value (MAX)
        # print(maxUtility)
        return chosenAction

    def getUtility(self, gameState, numAgents, pathMaxUtility, pathMinUtility):
        # check for leaf node
        if self.depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # get all legal moves (actions) for agent
        legalMoves = gameState.getLegalActions(self.index)
        # check for MAX agent (pacman)
        if self.index == 0:
            maxUtility = 0
            initialised = False
            # create next agent
            nextAgent = AlphaBetaAgent()
            nextAgent.evaluationFunction = self.evaluationFunction
            # check for ghosts
            if numAgents > 1:
                nextAgent.index = 1
                nextAgent.depth = self.depth
            else:
                # no ghosts, only pacman
                nextAgent.depth = self.depth - 1
            # determine max utility possible
            for action in legalMoves:
                successorGameState = gameState.generateSuccessor(
                    self.index, action)
                if initialised:
                    maxUtility = max(maxUtility, nextAgent.getUtility(
                        successorGameState, numAgents, pathMaxUtility, pathMinUtility))
                    if maxUtility > pathMinUtility:
                        # prune subtree
                        return maxUtility
                    # update pathMaxUtility appropriately
                    pathMaxUtility = max(pathMaxUtility, maxUtility)
                else:
                    maxUtility = nextAgent.getUtility(
                        successorGameState, numAgents, pathMaxUtility, pathMinUtility)
                    if maxUtility > pathMinUtility:
                        # prune subtree
                        return maxUtility
                    # update pathMaxUtility appropriately
                    pathMaxUtility = max(pathMaxUtility, maxUtility)
                    initialised = True
            # return utility value (MAX)
            return maxUtility
        else:
            # ghost (MIN) agent
            minUtility = 0
            initialised = False
            # create next agent
            nextAgent = AlphaBetaAgent()
            nextAgent.evaluationFunction = self.evaluationFunction
            # check for pacman (last ghost)
            if self.index + 1 == numAgents:
                nextAgent.depth = self.depth - 1
            else:
                # next agent is also ghost
                nextAgent.index = self.index + 1
                nextAgent.depth = self.depth
            # determine min utility possible
            for action in legalMoves:
                successorGameState = gameState.generateSuccessor(
                    self.index, action)
                if initialised:
                    minUtility = min(minUtility, nextAgent.getUtility(
                        successorGameState, numAgents, pathMaxUtility, pathMinUtility))
                    if minUtility < pathMaxUtility:
                        # prune subtree
                        return minUtility
                    # update pathMinUtility appropriately
                    pathMinUtility = min(minUtility, pathMinUtility)
                else:
                    minUtility = nextAgent.getUtility(
                        successorGameState, numAgents, pathMaxUtility, pathMinUtility)
                    if minUtility < pathMaxUtility:
                        # prune subtree
                        return minUtility
                    # update pathMinUtility appropriately
                    pathMinUtility = min(pathMinUtility, minUtility)
                    initialised = True
            # return utility value (MIN)
            return minUtility


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # check if game is over, or depth is 0
        if self.depth == 0 or gameState.isWin() or gameState.isLose():
            return None
        # store the number of agents (pacman and ghosts)
        numAgents = gameState.getNumAgents()
        legalMoves = gameState.getLegalActions(0)
        # create next agent
        nextAgent = ExpectimaxAgent()
        nextAgent.evaluationFunction = self.evaluationFunction
        # check for ghosts
        if numAgents > 1:
            nextAgent.index = 1
            nextAgent.depth = self.depth
        else:
            nextAgent.depth = self.depth - 1
        maxUtility = 0
        initialised = False
        for action in legalMoves:
            successorGameState = gameState.generateSuccessor(
                self.index, action)
            if initialised:
                temp = nextAgent.getUtility(successorGameState, numAgents)
                #print(temp, end=' ')
                #print(action, end=' ')
                if temp > maxUtility:
                    maxUtility = temp
                    chosenAction = action

                elif temp == maxUtility and chosenAction == 'Stop':
                    chosenAction = action
            else:
                maxUtility = nextAgent.getUtility(
                    successorGameState, numAgents)
                chosenAction = action
                initialised = True
                #print(maxUtility, end=' ')
                #print(action, end=' ')
        # print()
        # print(chosenAction)
        # return action resulting in maximum value (MAX)
        # if chosenAction == 'Stop':
        return chosenAction

    def getUtility(self, gameState, numAgents):
        # check for leaf node
        if self.depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # get all legal moves (actions) for agent
        legalMoves = gameState.getLegalActions(self.index)
        # check for MAX agent (pacman)
        if self.index == 0:
            maxUtility = 0
            initialised = False
            # create next agent
            nextAgent = ExpectimaxAgent()
            nextAgent.evaluationFunction = self.evaluationFunction
            # check for ghosts
            if numAgents > 1:
                nextAgent.index = 1
                nextAgent.depth = self.depth
            else:
                # no ghosts, only pacman
                nextAgent.depth = self.depth - 1
            # determine max utility possible
            for action in legalMoves:
                successorGameState = gameState.generateSuccessor(
                    self.index, action)
                if initialised:
                    maxUtility = max(maxUtility, nextAgent.getUtility(
                        successorGameState, numAgents))
                else:
                    maxUtility = nextAgent.getUtility(
                        successorGameState, numAgents)
                    initialised = True
            # return utility value (MAX)
            return maxUtility
        else:
            # ghost (Random) agent
            expectedUtility = 0
            # create next agent
            nextAgent = ExpectimaxAgent()
            nextAgent.evaluationFunction = self.evaluationFunction
            # check for pacman (last ghost)
            if self.index + 1 == numAgents:
                nextAgent.depth = self.depth - 1
            else:
                # next agent is also ghost
                nextAgent.index = self.index + 1
                nextAgent.depth = self.depth
            # determine expected utility value
            for action in legalMoves:
                successorGameState = gameState.generateSuccessor(
                    self.index, action)
                expectedUtility += nextAgent.getUtility(
                    successorGameState, numAgents)
            # return utility value (expectation (uniform))
            return expectedUtility / len(legalMoves)


def hasReachablePellet(pacX, pacY, currentGameState, foodGrid, foodList, capsuleList, wallGrid):
    que = util.PriorityQueue()
    dist = [[sys.maxsize for i in range(foodGrid.height)]
            for i in range(foodGrid.width)]
    que.push((pacX, pacY), 0)
    dist[pacX][pacY] = 0
    while not que.isEmpty():
        dist_yet = que.heap[0][0]
        (x, y) = que.pop()
        for (dx, dy) in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            x_new, y_new = (x + dx, y + dy)
            if 0 <= x_new < foodGrid.width and 0 <= y_new < foodGrid.height and not wallGrid[x_new][y_new]:
                dist_new = dist_yet + \
                    (1 if currentGameState.hasFood(x_new, y_new) else 0)
                if dist_new < dist[x_new][y_new]:
                    dist[x_new][y_new] = dist_new
                    que.push((x_new, y_new), dist_new)
    for capsule in capsuleList:
        (x, y) = capsule
        if dist[x][y] < len(foodList):
            return True
    return False


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # check for losing state, if True then penalise (-1,00,000)
    if currentGameState.isLose():
        return -1e12
    # determine pacman position
    pacmanPos = currentGameState.getPacmanPosition()
    # determine current game score
    gameScore = currentGameState.getScore()
    # determine walls position
    wallGrid = currentGameState.getWalls()
    # determine food position
    foodGrid = currentGameState.getFood()
    foodList = foodGrid.asList()
    foodExists = True
    if foodList == []:
        foodExists = False
    # determine ghosts positions, if any
    ghostsExist = True
    ghostStates = currentGameState.getGhostStates()
    if ghostStates == []:
        ghostsExist = False
    else:
        # determine ghost scared times and positions
        scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
        ghostList = [ghostState.getPosition() for ghostState in ghostStates]
    # determine capsule positions, if any
    capsulesExist = True
    capsuleList = currentGameState.getCapsules()
    if capsuleList == []:
        capsulesExist = False
    # determine number of food islands, if food exists
    if foodExists:
        # initialise number of islands
        num_islands = 0
        # initialise visited matrix
        visited = [[False for i in range(foodGrid.height)]
                   for i in range(foodGrid.width)]
        # initialise BFS queue
        BFS = util.Queue()
        for i in range(len(foodList)):
            if visited[foodList[i][0]][foodList[i][1]]:
                continue
            # insert initial food
            BFS.push((foodList[i][0], foodList[i][1]))
            visited[foodList[i][0]][foodList[i][1]] = True
            while not BFS.isEmpty():
                (x, y) = BFS.pop()
                # add food neighbours to queue
                if currentGameState.hasFood(x - 1, y) and not visited[x - 1][y]:
                    visited[x - 1][y] = True
                    BFS.push((x - 1, y))
                if currentGameState.hasFood(x + 1, y) and not visited[x + 1][y]:
                    visited[x + 1][y] = True
                    BFS.push((x + 1, y))
                if currentGameState.hasFood(x, y - 1) and not visited[x][y - 1]:
                    visited[x][y - 1] = True
                    BFS.push((x, y - 1))
                if currentGameState.hasFood(x, y + 1) and not visited[x][y + 1]:
                    visited[x][y + 1] = True
                    BFS.push((x, y + 1))
            # increment number of islands
            num_islands += 1
    else:
        # no food, no islands
        num_islands = 0.1
    # initialise distance matrix
    distance = [[-1 for i in range(foodGrid.height)]
                for i in range(foodGrid.width)]
    # initialise visited matrix
    visited = [[False for i in range(foodGrid.height)]
               for i in range(foodGrid.width)]
    if len(foodList) == 1:
        lastFood = foodList[0]
    else:
        lastFood = (-1, -1)
    # initialise BFS queue
    BFS = util.Queue()
    # insert initial state (distance=0)
    BFS.push((pacmanPos[0], pacmanPos[1]))
    distance[pacmanPos[0]][pacmanPos[1]] = 0
    visited[pacmanPos[0]][pacmanPos[1]] = True
    while not BFS.isEmpty():
        (x, y) = BFS.pop()
        # add neighbours to queue
        if not wallGrid[x - 1][y] and not visited[x - 1][y]:
            visited[x - 1][y] = True
            distance[x - 1][y] = distance[x][y] + 1
            if lastFood[0] != x - 1 or lastFood[1] != y:
                BFS.push((x - 1, y))
        if not wallGrid[x + 1][y] and not visited[x + 1][y]:
            visited[x + 1][y] = True
            distance[x + 1][y] = distance[x][y] + 1
            if lastFood[0] != x + 1 or lastFood[1] != y:
                BFS.push((x + 1, y))
        if not wallGrid[x][y - 1] and not visited[x][y - 1]:
            visited[x][y - 1] = True
            distance[x][y - 1] = distance[x][y] + 1
            if lastFood[0] != x or lastFood[1] != y - 1:
                BFS.push((x, y - 1))
        if not wallGrid[x][y + 1] and not visited[x][y + 1]:
            visited[x][y + 1] = True
            distance[x][y + 1] = distance[x][y] + 1
            if lastFood[0] != x or lastFood[1] != y + 1:
                BFS.push((x, y + 1))
    # BFS complete, distance matrix found
    que = util.PriorityQueue()
    dist = [[sys.maxsize for i in range(foodGrid.height)]
            for i in range(foodGrid.width)]
    que.push((pacmanPos[0], pacmanPos[1]), 0)
    dist[pacmanPos[0]][pacmanPos[1]] = 0
    while not que.isEmpty():
        dist_yet = que.heap[0][0]
        (x, y) = que.pop()
        for (dx, dy) in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            x_new, y_new = (x + dx, y + dy)
            if not wallGrid[x_new][y_new]:
                dist_new = dist_yet + (1 if currentGameState.hasFood(x_new, y_new) else 0)
                if dist_new < dist[x_new][y_new]:
                    dist[x_new][y_new] = dist_new
                    que.update((x_new, y_new), dist_new)
    capsuleDist = [dist[capsule[0]][capsule[1]] for capsule in capsuleList if dist[capsule[0]][capsule[1]] != sys.maxsize]
    ghostDist = [dist[int(ghost[0])][int(ghost[1])] for ghost in ghostList if dist[int(ghost[0])][int(ghost[1])] != sys.maxsize]
    # determine actual distances to food (if it exists)
    if foodExists:
        foodDistance = [distance[food[0]][food[1]] for food in foodList]
    # CASE 1: Ghosts exist
    if ghostsExist:
        # determine actual distances to ghosts
        ghostDistance = [distance[int(ghost[0])][int(ghost[1])] for ghost in ghostList]
        # ASSUMPTION: maximum number of ghosts = 4 (any layout)
        # determine ghost with maximum scared timer
        # tie break using distance
        maxTime = 0
        ghostIndex = -1
        for i in range(len(scaredTimes)):
            if ghostDistance[i] == -1:
                continue
            if scaredTimes[i] > maxTime:
                maxTime = scaredTimes[i]
                ghostIndex = i
            elif scaredTimes[i] == maxTime and maxTime != 0:
                if ghostDistance[i] < ghostDistance[ghostIndex]:
                    ghostIndex = i
        # determine distances to reachable capsules
        capsuleDistance = [distance[capsule[0]][capsule[1]] for capsule in capsuleList if distance[capsule[0]][capsule[1]] != -1]
        # check if pacman is winning (and no capsule or scared ghosts exist)
        if currentGameState.isWin() and ghostIndex == -1 and len(capsuleDistance) == 0:
            return gameScore * 1e6 + 1e6
        # count dangerous ghosts (distance < 3 and scaredTimer < distance)
        dangerCount = 0
        for i in range(len(ghostDistance)):
            if ghostDistance[i] < 3 and scaredTimes[i] < ghostDistance[i]:
                dangerCount += 1
        # count number of legal actions in current state, subtract 1 for STOP
        actionCount = len(currentGameState.getLegalActions()) - 1
        # penalise state, if pacman is getting cornered
        if actionCount <= dangerCount and not currentGameState.isWin():
            # play not worth it
            return -1e10
        # else, don't be afraid of ghosts
        # check for reachable scared ghost
        if ghostIndex != -1 and (min(ghostDist) < len(foodList)):
            # reward pacman for being near to a scared ghost (200 points)
            if currentGameState.isWin():
                #changed 1e6 to 1e5. !!!NEED TO CHECK
                return (gameScore - 510) * 1e6 + 1e5 + 10
            else:
                # + 1e7 / scaredTimes[ghostIndex]
                return 2e7 / ghostDistance[ghostIndex] + (gameScore + 10) * 1e6 + 100000 / num_islands + 1e7
        elif ghostIndex != -1:
            # if currentGameState.isWin():
            #     return (gameScore - 510) * 1e6 + 1e5 + 10
            return 2e7 / ghostDistance[ghostIndex] + (gameScore + 10) * 1e6 + 100000 / num_islands + 1e7
        # no reachable ghosts are scared, check for capsules (hunt ghosts)

        if capsulesExist and len(capsuleDistance) != 0 and (min(capsuleDist) < len(foodList) or len(foodList) == 0):
            # reward pacman for being near to a capsule
            if currentGameState.isWin():
                return (gameScore - 510) * 1e6 + 1e5 + 10
            else:
                return 100000 / min(capsuleDistance) + (gameScore) * 1e6 + 100000 / num_islands + 1e7
        # no capsule reachable
        # reward near-food locations (min foodDistance) and food-eating (high score)
        return 5 / min(foodDistance) + gameScore * 1e6 + 100000 / num_islands
    # CASE 2: Ghosts do not exist
    # check for winning state, if True then reward
    if currentGameState.isWin():
        return gameScore * 10000 + 100000 / num_islands
    # else, not a winning state (foodExists)
    # reward near-food locations (min foodDistance) and food-eating (high score)
    return 1000 / min(foodDistance) + gameScore * 10000 + 100000 / num_islands


# Abbreviation
better = betterEvaluationFunction
