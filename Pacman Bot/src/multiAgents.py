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
import random, util

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
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # check if it is a winning state, return 100 (reward) if True
        if successorGameState.isWin():
            return 100
        # determine ghost position
        ghostList = [ghostState.getPosition() for ghostState in newGhostStates]
        ghostDistance = [manhattanDistance(newPos, ghost) for ghost in ghostList]
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
            successorGameState = gameState.generateSuccessor(self.index, action)
            if initialised:
                temp = nextAgent.getUtility(successorGameState, numAgents)
                if temp > maxUtility:
                    maxUtility = temp
                    chosenAction = action
            else:
                maxUtility = nextAgent.getUtility(successorGameState, numAgents)
                chosenAction = action
                initialised = True
        # return action resulting in maximum value (MAX)
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
                successorGameState = gameState.generateSuccessor(self.index, action)
                if initialised:
                    maxUtility = max(maxUtility, nextAgent.getUtility(successorGameState, numAgents))
                else:
                    maxUtility = nextAgent.getUtility(successorGameState, numAgents)
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
                successorGameState = gameState.generateSuccessor(self.index, action)
                if initialised:
                    minUtility = min(minUtility, nextAgent.getUtility(successorGameState, numAgents))
                else:
                    minUtility = nextAgent.getUtility(successorGameState, numAgents)
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
            successorGameState = gameState.generateSuccessor(self.index, action)
            if initialised:
                temp = nextAgent.getUtility(successorGameState, numAgents, pathMaxUtility, pathMinUtility)
                if temp > maxUtility:
                    maxUtility = temp
                    chosenAction = action
                # update pathMaxUtility appropriately
                pathMaxUtility = max(pathMaxUtility, maxUtility)
            else:
                maxUtility = nextAgent.getUtility(successorGameState, numAgents, pathMaxUtility, pathMinUtility)
                chosenAction = action
                # update pathMaxUtility appropriately
                pathMaxUtility = max(pathMaxUtility, maxUtility)
                initialised = True
        # return action resulting in maximum value (MAX)
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
                successorGameState = gameState.generateSuccessor(self.index, action)
                if initialised:
                    maxUtility = max(maxUtility, nextAgent.getUtility(successorGameState, numAgents, pathMaxUtility, pathMinUtility))
                    if maxUtility > pathMinUtility:
                        # prune subtree
                        return maxUtility
                    # update pathMaxUtility appropriately
                    pathMaxUtility = max(pathMaxUtility, maxUtility)
                else:
                    maxUtility = nextAgent.getUtility(successorGameState, numAgents, pathMaxUtility, pathMinUtility)
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
                successorGameState = gameState.generateSuccessor(self.index, action)
                if initialised:
                    minUtility = min(minUtility, nextAgent.getUtility(successorGameState, numAgents, pathMaxUtility, pathMinUtility))
                    if minUtility < pathMaxUtility:
                        # prune subtree
                        return minUtility
                    # update pathMinUtility appropriately
                    pathMinUtility = min(minUtility, pathMinUtility)
                else:
                    minUtility = nextAgent.getUtility(successorGameState, numAgents, pathMaxUtility, pathMinUtility)
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
            successorGameState = gameState.generateSuccessor(self.index, action)
            if initialised:
                temp = nextAgent.getUtility(successorGameState, numAgents)
                if temp > maxUtility:
                    maxUtility = temp
                    chosenAction = action
                elif temp == maxUtility and chosenAction == 'Stop':
                    chosenAction = action
            else:
                maxUtility = nextAgent.getUtility(successorGameState, numAgents)
                chosenAction = action
                initialised = True
        # return action resulting in maximum value (MAX)
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
                successorGameState = gameState.generateSuccessor(self.index, action)
                if initialised:
                    maxUtility = max(maxUtility, nextAgent.getUtility(successorGameState, numAgents))
                else:
                    maxUtility = nextAgent.getUtility(successorGameState, numAgents)
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
                successorGameState = gameState.generateSuccessor(self.index, action)
                expectedUtility += nextAgent.getUtility(successorGameState, numAgents)
            # return utility value (expectation (uniform))
            return expectedUtility / len(legalMoves)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: <write something here so we know what you did>
    """
    # Basically, we look out for ghosts (normal/scared), capsules and food items and try to maximize
    # score, and avoiding losing
    "*** YOUR CODE HERE ***"
    # check for losing state, if True then penalise
    if currentGameState.isLose():
        return -1e12
    # constant
    INF = 1000000000
    # determine pacman state
    pacmanState = currentGameState.getPacmanState()
    pacmanPos = pacmanState.getPosition()
    pacmanDirection = pacmanState.getDirection()
    # determine walls position
    wallGrid = currentGameState.getWalls()
    startPos = (pacmanPos[0], pacmanPos[1])
    if currentGameState.isWin():
        if pacmanDirection == 'West':
            startPos = (pacmanPos[0] + 1, pacmanPos[1])
        elif pacmanDirection == 'East':
            startPos = (pacmanPos[0] - 1, pacmanPos[1])
        elif pacmanDirection == 'North':
            startPos = (pacmanPos[0], pacmanPos[1] - 1)
        elif pacmanDirection == 'South':
            startPos = (pacmanPos[0], pacmanPos[1] + 1)
    # determine current game score
    gameScore = currentGameState.getScore() + 2000
    # determine food position
    foodGrid = currentGameState.getFood()
    # determine food count
    foodCount = currentGameState.getNumFood()
    # determine ghosts states
    ghostStates = currentGameState.getGhostStates()
    # divide ghosts into scared and normal
    scaredGhostList = set()
    normalGhostList = set()
    ghostTime = -1
    for ghostState in ghostStates:
        if ghostState.scaredTimer > 0:
            ghostTime = ghostState.scaredTimer
            pos = ghostState.getPosition()
            scaredGhostList.add((int(pos[0]), int(pos[1])))
        else:
            pos = ghostState.getPosition()
            normalGhostList.add((int(pos[0]), int(pos[1])))
    # determine capsule positions
    capsuleList = set(currentGameState.getCapsules())
    # perform partial BFS from pacman and determine reachability
    distance = {}
    visited = set()
    # initialise BFS queue
    BFS = util.Queue()
    # insert initial state (distance=0)
    BFS.push((startPos[0], startPos[1]))
    distance[(startPos[0], startPos[1])] = 0
    visited.add((startPos[0], startPos[1]))
    visited.add((pacmanPos[0], pacmanPos[1]))
    # initialise flags
    foodNotEmpty = False
    scaredGhostsNotEmpty = False
    normalGhostsNotEmpty = False
    capsulesNotEmpty = False
    # initialise distances
    minFoodDistance = INF
    minScaredGhostDistance = INF
    minNormalGhostDistance = INF
    minCapsuleDistance = INF
    ghostCounter = 0
    while not BFS.isEmpty():
        (x, y) = BFS.pop()
        if foodGrid[x][y]:
            foodNotEmpty = True
            minFoodDistance = min(minFoodDistance, distance[(x, y)])
        if (x, y) in scaredGhostList:
            scaredGhostsNotEmpty = True
            minScaredGhostDistance = min(minScaredGhostDistance, distance[(x, y)])
        if (x, y) in normalGhostList:
            if distance[(x, y)] < 5:
                ghostCounter += 1
            normalGhostsNotEmpty = True
            minNormalGhostDistance = min(minNormalGhostDistance, distance[(x, y)])
        if (x, y) in capsuleList:
            capsulesNotEmpty = True
            minCapsuleDistance = min(minCapsuleDistance, distance[(x, y)])
        if distance[(x, y)] >= 5:
            if scaredGhostsNotEmpty:
                break
            if len(scaredGhostList) == 0:
                if len(normalGhostList) == 0:
                    if currentGameState.isWin():
                        break
                    elif foodNotEmpty:
                        break
                else:
                    if len(capsuleList) == 0:
                        if normalGhostsNotEmpty:
                            if currentGameState.isWin():
                                break
                            elif foodNotEmpty:
                                break
                    else:
                        if normalGhostsNotEmpty:
                            if currentGameState.isWin():
                                if capsulesNotEmpty:
                                    break
                            else:
                                if capsulesNotEmpty:
                                    if foodCount == 1:
                                        break
                                    elif foodNotEmpty:
                                        break
            else:
                if distance[(x, y)] > 2 * ghostTime:
                    scaredGhostList = set()
        # add neighbours to queue
        if not wallGrid[x - 1][y] and (x - 1, y) not in visited:
            visited.add((x - 1, y))
            distance[(x - 1, y)] = distance[(x, y)] + 1
            if (x - 1, y) not in normalGhostList and (not foodGrid[x - 1][y] or foodCount != 1):
                BFS.push((x - 1, y))
            else:
                if foodGrid[x - 1][y]:
                    foodNotEmpty = True
                    minFoodDistance = min(minFoodDistance, distance[(x - 1, y)])
                if (x - 1, y) in scaredGhostList:
                    scaredGhostsNotEmpty = True
                    minScaredGhostDistance = min(minScaredGhostDistance, distance[(x - 1, y)])
                if (x - 1, y) in normalGhostList:
                    if distance[(x - 1, y)] < 5:
                        ghostCounter += 1
                    normalGhostsNotEmpty = True
                    minNormalGhostDistance = min(minNormalGhostDistance, distance[(x - 1, y)])
                if (x - 1, y) in capsuleList:
                    capsulesNotEmpty = True
                    minCapsuleDistance = min(minCapsuleDistance, distance[(x - 1, y)])
        if not wallGrid[x + 1][y] and (x + 1, y) not in visited:
            visited.add((x + 1, y))
            distance[(x + 1, y)] = distance[(x, y)] + 1
            if (x + 1, y) not in normalGhostList and (not foodGrid[x + 1][y] or foodCount != 1):
                BFS.push((x + 1, y))
            else:
                if foodGrid[x + 1][y]:
                    foodNotEmpty = True
                    minFoodDistance = min(minFoodDistance, distance[(x + 1, y)])
                if (x + 1, y) in scaredGhostList:
                    scaredGhostsNotEmpty = True
                    minScaredGhostDistance = min(minScaredGhostDistance, distance[(x + 1, y)])
                if (x + 1, y) in normalGhostList:
                    if distance[(x + 1, y)] < 5:
                        ghostCounter += 1
                    normalGhostsNotEmpty = True
                    minNormalGhostDistance = min(minNormalGhostDistance, distance[(x + 1, y)])
                if (x + 1, y) in capsuleList:
                    capsulesNotEmpty = True
                    minCapsuleDistance = min(minCapsuleDistance, distance[(x + 1, y)])
        if not wallGrid[x][y - 1] and (x, y - 1) not in visited:
            visited.add((x, y - 1))
            distance[(x, y - 1)] = distance[(x, y)] + 1
            if (x, y - 1) not in normalGhostList and (not foodGrid[x][y - 1] or foodCount != 1):
                BFS.push((x, y - 1))
            else:
                if foodGrid[x][y - 1]:
                    foodNotEmpty = True
                    minFoodDistance = min(minFoodDistance, distance[(x, y - 1)])
                if (x, y - 1) in scaredGhostList:
                    scaredGhostsNotEmpty = True
                    minScaredGhostDistance = min(minScaredGhostDistance, distance[(x, y - 1)])
                if (x, y - 1) in normalGhostList:
                    if distance[(x, y - 1)] < 5:
                        ghostCounter += 1
                    normalGhostsNotEmpty = True
                    minNormalGhostDistance = min(minNormalGhostDistance, distance[(x, y - 1)])
                if (x, y - 1) in capsuleList:
                    capsulesNotEmpty = True
                    minCapsuleDistance = min(minCapsuleDistance, distance[(x, y - 1)])
        if not wallGrid[x][y + 1] and (x, y + 1) not in visited:
            visited.add((x, y + 1))
            distance[(x, y + 1)] = distance[(x, y)] + 1
            if (x, y + 1) not in normalGhostList and (not foodGrid[x][y + 1] or foodCount != 1):
                BFS.push((x, y + 1))
            else:
                if foodGrid[x][y + 1]:
                    foodNotEmpty = True
                    minFoodDistance = min(minFoodDistance, distance[(x, y + 1)])
                if (x, y + 1) in scaredGhostList:
                    scaredGhostsNotEmpty = True
                    minScaredGhostDistance = min(minScaredGhostDistance, distance[(x, y + 1)])
                if (x, y + 1) in normalGhostList:
                    if distance[(x, y + 1)] < 5:
                        ghostCounter += 1
                    normalGhostsNotEmpty = True
                    minNormalGhostDistance = min(minNormalGhostDistance, distance[(x, y + 1)])
                if (x, y + 1) in capsuleList:
                    capsulesNotEmpty = True
                    minCapsuleDistance = min(minCapsuleDistance, distance[(x, y + 1)])
    # BFS complete, distance matrix found
    riskFactor = 1
    actionCount = len(currentGameState.getLegalActions()) - 1
    if actionCount <= ghostCounter and not currentGameState.isWin():
        riskFactor = 10
    # check if scared ghosts exist
    if scaredGhostsNotEmpty:
        if not currentGameState.isWin():
            # reward pacman for being closer to a scared ghost
            return (1e8 / minScaredGhostDistance + gameScore * 1e6)/riskFactor
        else:
            # pacman should go for ghost, give low reward
            return (gameScore * 1e3)/riskFactor
    # scared ghosts do not exist, check if normal ghosts exist
    if normalGhostsNotEmpty:
        if not currentGameState.isWin():
            if not capsulesNotEmpty:
                if foodNotEmpty:
                    # no reachable capsules exist, so just eat food
                    return (gameScore * 1e6 + 1e6 / minFoodDistance)/riskFactor
                else:
                    # ghosts are probably blocking path to food, move towards a random ghost
                    return (gameScore * 1e6 + 1e3 / random.choice([1, 2, 3, 4, 5]))/riskFactor
            else:
                # capsules exist, reward pacman for being closer to a capsule
                if foodNotEmpty and foodCount > 1:
                    return (gameScore * 1e6 + 1e6 / minCapsuleDistance + 1e5 / minFoodDistance)/riskFactor
                else:
                    return (gameScore * 1e6 + 1e6 / minCapsuleDistance)/riskFactor
        else:
            if not capsulesNotEmpty:
                # reward pacman for finishing game
                return (gameScore * 1e6)/riskFactor
            else:
                # pacman should go for capsule, give low reward
                return (gameScore * 1e3)/riskFactor
    # normal ghosts do not exist either
    # reward is proportional to game score (if win) and game score plus proximity to food otherwise
    if currentGameState.isWin():
        return (gameScore * 1e6)/riskFactor
    else:
        # assuming that game is winnable, foodList is non-empty
        return (gameScore * 1e6 + 1e6 / minFoodDistance)/riskFactor

# Abbreviation
better = betterEvaluationFunction
