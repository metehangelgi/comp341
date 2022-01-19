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
        capsules=successorGameState.getCapsules()
        foods = newFood.asList()
        eval = successorGameState.getScore()
        minInf=-1 * float("inf")
        maxInf=float("inf")
        if action == "Stop": #action should not be stop to get rid of stuck
            eval= minInf
        else:
            if len(foods) != 0:
                foodDist = maxInf
                for food in foods:
                    foodDist = min(manhattanDistance(food, newPos), foodDist)

            if len(capsules) > 0:
                capsuleDist = maxInf
                for capsule in capsules:
                    capsuleDist = min(manhattanDistance(newPos, capsule), capsuleDist)

            if len(newGhostStates) != 0:
                for newGhostStateIndex in range(len(newGhostStates)):
                    ghostScareTime=newScaredTimes[newGhostStateIndex]
                    ghostDistance = manhattanDistance(newPos, newGhostStates[newGhostStateIndex].getPosition())
                    # sometimes I got error since ghosts are also moving
                    if ghostDistance == 0:
                        ghostDistance = 0.0001

                    if ghostScareTime > 0 and (newScaredTimes[newGhostStateIndex] - ghostDistance) > 0:
                            eval = eval + 300 / ghostDistance
                    else:
                        eval = eval - 300 / ghostDistance
                        if len(capsules) != 0:
                            eval = eval + 150 / capsuleDist

            eval = eval - len(foods) * 100
            eval = eval - len(capsules) * 100
            if len(foods) != 0:
                eval = eval + 100 / foodDist

        return eval
        #return successorGameState.getScore()

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
        currentDepth=0
        currIndex=0
        eval,moveTo = self.minimaxValue(gameState, currentDepth, currIndex)
        return moveTo
        #util.raiseNotDefined()

    def minimaxValue(self, gameState, currentDepth, currIndex):
        moveTo = None
        if gameState.isWin() or gameState.isLose() or currentDepth >= self.depth:
            return self.evaluationFunction(gameState),moveTo
        elif currIndex == self.index:
            return self.maxValue(gameState, currentDepth,currIndex)
        else:
            return self.minValue(gameState, currentDepth, currIndex)

    def maxValue(self, gameState, currentDepth, currIndex):
        maxVal = -1 * float("inf")
        maxAction = None
        for action in gameState.getLegalActions(currIndex):
            eval,moveTo = self.minimaxValue(gameState.generateSuccessor(currIndex, action), currentDepth, currIndex+1)
            if eval>maxVal:
                maxVal = eval
                maxAction=action
        return maxVal,maxAction

    def minValue(self,gameState, currentDepth, currIndex):
        minVal = float("inf")
        minAction = None
        if currIndex == gameState.getNumAgents() - 1:
            currentDepth += 1
            nextIndex = 0
        else:
            nextIndex = currIndex + 1

        for action in gameState.getLegalActions(currIndex):
            eval,moveTo = self.minimaxValue(gameState.generateSuccessor(currIndex, action), currentDepth, nextIndex)
            if eval<minVal:
                minVal = eval
                minAction=action
        return minVal,minAction

        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        currentDepth = 0
        currIndex = 0
        alpha = -1 * float("inf")
        beta = float("inf")
        eval, moveTo = self.minimaxValue(gameState, currentDepth, currIndex,alpha, beta)
        return moveTo
        #util.raiseNotDefined()

    def minimaxValue(self, gameState, currentDepth, currIndex,alpha, beta):
        moveTo = None
        if gameState.isWin() or gameState.isLose() or currentDepth >= self.depth:
            return self.evaluationFunction(gameState),moveTo
        elif currIndex == self.index:
            return self.maxValue(gameState, currentDepth,currIndex,alpha, beta)
        else:
            return self.minValue(gameState, currentDepth, currIndex,alpha, beta)

    def maxValue(self, gameState, currentDepth, currIndex, alpha, beta):
        maxVal = -1 * float("inf")
        maxAction = None
        for action in gameState.getLegalActions(currIndex):
            eval,moveTo = self.minimaxValue(gameState.generateSuccessor(currIndex, action), currentDepth, currIndex+1,alpha, beta)
            if eval>maxVal:
                maxVal = eval
                maxAction=action
            if beta < maxVal:
                return maxVal,maxAction
            alpha=max(alpha,maxVal)

        return maxVal,maxAction

    def minValue(self,gameState, currentDepth, currIndex, alpha, beta):
        minVal = float("inf")
        minAction = None
        if currIndex == gameState.getNumAgents() - 1:
            currentDepth += 1
            nextIndex = 0
        else:
            nextIndex = currIndex + 1

        for action in gameState.getLegalActions(currIndex):
            eval,moveTo = self.minimaxValue(gameState.generateSuccessor(currIndex, action), currentDepth, nextIndex,alpha, beta)
            if eval<minVal:
                minVal = eval
                minAction=action
            if minVal < alpha:
                return minVal,minAction
            beta = min(beta, minVal)

        return minVal,minAction

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
        currentDepth = 0
        currIndex = 0
        eval, moveTo = self.expectimax(gameState, currentDepth, currIndex)
        return moveTo
        # util.raiseNotDefined()

    def expectimax(self, gameState, currentDepth, currIndex):
        moveTo = None
        if gameState.isWin() or gameState.isLose() or currentDepth >= self.depth:
            return self.evaluationFunction(gameState), moveTo
        elif currIndex == self.index:
            return self.maxValue(gameState, currentDepth, currIndex)
        else:
            return self.minValue(gameState, currentDepth, currIndex)

    def maxValue(self, gameState, currentDepth, currIndex):
        maxVal = -1 * float("inf")
        maxAction = None
        for action in gameState.getLegalActions(currIndex):
            eval, moveTo = self.expectimax(gameState.generateSuccessor(currIndex, action), currentDepth,
                                             currIndex + 1)
            if eval > maxVal:
                maxVal = eval
                maxAction = action
        return maxVal, maxAction

    def minValue(self, gameState, currentDepth, currIndex):
        val = 0
        NoneAction = None
        if currIndex == gameState.getNumAgents() - 1:
            currentDepth += 1
            nextIndex = 0
        else:
            nextIndex = currIndex + 1

        for action in gameState.getLegalActions(currIndex):
            eval, NoneAction = self.expectimax(gameState.generateSuccessor(currIndex, action), currentDepth, nextIndex)
            val = val + (1 / float(len(gameState.getLegalActions(currIndex)))) * eval
        return val, NoneAction


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
        I used an evaluation function very similar to the one I used for the first evaluation function. 
        This time I got CurrentGameState because we couldn't get SuccessorGameState.
        I just changed the weight of the scared ghost chase feature(distance to the closest ghost if it is scared)
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    capsules = currentGameState.getCapsules()
    foods = newFood.asList()
    eval = currentGameState.getScore()

    maxInf = float("inf")

    if len(foods) != 0:
        foodDist = maxInf
        for food in foods:
            foodDist = min(manhattanDistance(food, newPos), foodDist)

    if len(capsules) > 0:
        capsuleDist = maxInf
        for capsule in capsules:
            capsuleDist = min(manhattanDistance(newPos, capsule), capsuleDist)

    if len(newGhostStates) != 0:
        for newGhostStateIndex in range(len(newGhostStates)):
            ghostScareTime = newScaredTimes[newGhostStateIndex]
            ghostDistance = manhattanDistance(newPos, newGhostStates[newGhostStateIndex].getPosition())
            # sometimes I got error since ghosts are also moving
            if ghostDistance == 0:
                ghostDistance = 0.0001

            if ghostScareTime > 0 and (newScaredTimes[newGhostStateIndex] - ghostDistance) > 0:
                eval = eval + 1000 / ghostDistance
            else:
                eval = eval - 300 / ghostDistance
                if len(capsules) != 0:
                    eval = eval + 150 / capsuleDist

    eval = eval - len(foods) * 100
    eval = eval - len(capsules) * 100
    if len(foods) != 0:
        eval = eval + 100 / foodDist

    return eval

# Abbreviation
better = betterEvaluationFunction
