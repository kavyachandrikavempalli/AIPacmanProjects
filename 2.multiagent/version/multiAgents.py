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
        print(newPos)
        print(newFood)
        print(newGhostStates)
        print(newScaredTimes)
        score=0
        if successorGameState.isWin()==1:
            return maxscore
        elif successorGameState.isLose()==1:
            return minscore
        food_dist=[]
        for i in newFood.asList():
            temp_dist=manhattanDistance(newPos, i)
            food_dist.append(temp_dist)
        d_ghost=[]
        for i in newGhostStates:
            ghost_pos=i.getPosition()
            g_d=manhattanDistance(newPos, ghost_pos)
            d_ghost.append(g_d)
            
        if min(d_ghost)>0 and min(d_ghost)<=2:
            score-=1000
        if min(food_dist)<1:
            score+=1000
        "*** YOUR CODE HERE ***"
        return score
        #return successorGameState.getScore()

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
        minmax,minmaxAction=self.goDeep(gameState,self.depth,0,gameState.getNumAgents(),0,float('-inf'),float('inf'))

        return minmaxAction
        

    def goDeep(self,gameState, toDepth,reached,numOfAgents,reachedDepth,alpha,beta):
        if reachedDepth==toDepth or len(gameState.getLegalActions(reached%numOfAgents))==0:
            minmax=self.evaluationFunction(gameState))
            return minmax,''
        minmax=0.0
        minmaxAction=''
        if(reached%numOfAgents==0):
            minmax=float('-inf')
        else:
            minmax=float('inf')

        
        for action in gameState.getLegalActions(reached%numOfAgents):
            newGameState=gameState.generateSuccessor(reached%numOfAgents, action)
            if(reached%numOfAgents==0):
                newMinmax,newMinmaxAction=self.goDeep(newGameState,toDepth,reached+1,numOfAgents,reachedDepth,alpha,beta)
                if newMinmax>beta:
                    return newMinmax,minmaxAction
                elif minmax<=newMinmax:
                    minmax=newMinmax
                    alpha=max(alpha,minmax)
                    minmaxAction=action
            else:
                add=0
                if(reached%numOfAgents==numOfAgents-1):
                    add=1
                newMinmax,newMinmaxAction=self.goDeep(newGameState,toDepth,reached+1,numOfAgents,reachedDepth+add,alpha,beta)
                if newMinmax<alpha:
                    return newMinmax,minmaxAction
                elif minmax>=newMinmax:
                    minmax=newMinmax
                    beta=min(beta,minmax)
                    minmaxAction=action
        return minmax,minmaxAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
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
        util.raiseNotDefined()

