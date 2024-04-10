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
        maxscore=50000000
        minscore=-50000000
        stopPenalty=0
        if action=='Stop':
            stopPenalty=1000
        if successorGameState.isWin()==1:
            return maxscore
        elif successorGameState.isLose()==1:
            return minscore
        food_dist=[]
        for i in newFood.asList():
            temp_dist=manhattanDistance(newPos, i)
            food_dist.append(temp_dist)
        if len(food_dist)==0:
            return 0
        dist_score=min(food_dist)
        d_ghost=[]
        for i in newGhostStates:
            ghost_pos=i.getPosition()
            g_d=manhattanDistance(newPos, ghost_pos)
            d_ghost.append(g_d)
        ghost_score=min(d_ghost)
        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()+1/dist_score-1/ghost_score-stopPenalty

def scoreEvaluationFunction(currentGameState):
    """2
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
        agent = 0
        depth = 0

        def max(agent,gameState, depth):
            optimalScore=float('-inf')
            actionsAllowed=gameState.getLegalActions(agent)
            for action in actionsAllowed:
                successorState=gameState.generateSuccessor(agent,action)
                val,act = minmax(successorState,depth+1)
                if val>optimalScore:
                    optimalScore = val
                    bestAction = action
            return optimalScore,bestAction
        
        def min(agent,gameState, depth):
            optimalScore=float('inf')
            actionsAllowed=gameState.getLegalActions(agent)
            for action in actionsAllowed:
                successorState=gameState.generateSuccessor(agent,action)
                val,act=minmax(successorState,depth+1)
                if val < optimalScore:
                    optimalScore = val
                    bestAction = action
            return optimalScore,bestAction

        def minmax(gameState,depth):
            if depth==self.depth*gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), ''
            agent = depth%gameState.getNumAgents()
            if agent==0:
                return max(agent,gameState, depth)
            else:
                return min(agent,gameState, depth)
                
        val, action = minmax(gameState, depth)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        agent = 0
        depth = 0
        alpha = float('-inf')
        beta = float('inf')

        def max(agent,gameState, depth,alpha,beta):
            optimalScore=float('-inf')
            actionsAllowed=gameState.getLegalActions(agent)
            for action in actionsAllowed:
                successorState=gameState.generateSuccessor(agent,action)
                val,act = alphabeta(successorState,depth+1,alpha,beta)
                if val>optimalScore:
                    optimalScore = val
                    bestAction = action
                if optimalScore > alpha:
                    alpha = optimalScore
                if alpha > beta:
                    break
            return optimalScore,bestAction
        
        def min(agent,gameState, depth,alpha,beta):
            optimalScore=float('inf')
            actionsAllowed=gameState.getLegalActions(agent)
            for action in actionsAllowed:
                successorState=gameState.generateSuccessor(agent,action)
                val,act=alphabeta(successorState,depth+1,alpha,beta)
                if val < optimalScore:
                    optimalScore = val
                    bestAction = action
                if optimalScore < beta:
                    beta = optimalScore
                if alpha > beta:
                    break
            return optimalScore,bestAction

        def alphabeta(gameState,depth,alpha,beta):
            if depth==self.depth*gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), ''
            agent = depth%gameState.getNumAgents()
            if agent==0:
                return max(agent,gameState, depth,alpha,beta)
            else:
                return min(agent,gameState, depth,alpha,beta)
                
        val, action = alphabeta(gameState, depth,alpha,beta)
        return action

        util.raiseNotDefined()

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
        agent = 0
        depth = 0

        def max_node(agent,gameState, depth):
            optimalScore=float('-inf')
            actionsAllowed=gameState.getLegalActions(agent)
            for action in actionsAllowed:
                successorState=gameState.generateSuccessor(agent,action)
                val,act = expectimax(successorState,depth+1)
                if val>optimalScore:
                    optimalScore = val
                    bestAction = action
            return optimalScore,bestAction
        
        def expect_node(agent,gameState, depth):
            #optimalScore=float('inf')
            overallScore=0
            actionsAllowed=gameState.getLegalActions(agent)
            for action in actionsAllowed:
                successorState=gameState.generateSuccessor(agent,action)
                val,act=expectimax(successorState,depth+1)
                overallScore+=val
                # if val < optimalScore:
                #     optimalScore = val
                bestAction = action
            return overallScore/len(actionsAllowed),bestAction

        def expectimax(gameState,depth):
            if depth==self.depth*gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), ''
            agent = depth%gameState.getNumAgents()
            if agent==0:
                return max_node(agent,gameState, depth)
            else:
                return expect_node(agent,gameState, depth)
                
        val, action = expectimax(gameState, depth)
        return action
        util.raiseNotDefined()

