from typing import List, Tuple, Set, Dict, Optional
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, os
from capture import GameState, noisyDistance
from game import Directions, Actions, AgentState, Agent
from util import nearestPoint
from functools import lru_cache

# Base folder for file paths
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

from lib_piglet.utils.pddl_solver import pddl_solver
from lib_piglet.domains.pddl import pddl_state
from lib_piglet.utils.pddl_parser import Action

# Distance thresholds for PDDL predicates
CLOSE_DISTANCE = 4
MEDIUM_DISTANCE = 15
LONG_DISTANCE = 25


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='MixedAgent', second='MixedAgent', **args):
    """
    This function creates a team with one attacker and one defender.
    """
    return [
        eval(first)(firstIndex, role='attacker'),
        eval(second)(secondIndex, role='defender'),
    ]


##########
# Agents #
##########

class MixedAgent(CaptureAgent):
    """
    Improved hybrid agent with more aggressive attacking strategy.
    """
    
    # Q-Learning weights - More aggressive settings
    QLWeights = {
        "offensiveWeights": {
        'closest-food': -6,           
        'bias': -1,
        '#-of-ghosts-1-step-away': -60,  
        'successorScore': 150,         
        'chance-return-food': 80,        
        'depth': 3,                      
        'reverse': -5,
        'stop': -150,                 
        },
        "defensiveWeights": {
        'numInvaders': -1200,            
        'onDefense': 100,                
        'teamDistance': -2,
        'invaderDistance': -100,        
        'blockingEscape': 80,           
        'distanceToFood': -8,           
        'stop': -150,                    
        'reverse': -5,
        },
        "escapeWeights": {
        'distanceToHome': -150,          
        'enemyDistance': 25,            
        'veryCloseGhost': -800,         
        'carryingFood': 25,              
        'onDefense': 300,                
        'stop': -150,                   
        'reverse': -8,                  
        }
    }

    QLWeightsFile = BASE_FOLDER + '/QLWeightsMyTeam.txt'
    CURRENT_ACTION = {}

    def __init__(self, index, isRed=None, role='attacker', **kwargs):
        super().__init__(index)
        self.isRed = isRed
        self.role = role

    def registerInitialState(self, gameState: GameState):
        self.pddl_solver = pddl_solver(BASE_FOLDER + '/myTeam.pddl')
        self.highLevelPlan: List[Tuple[Action, pddl_state]] = []
        self.currentNegativeGoalStates = []
        self.currentPositiveGoalStates = []
        self.currentActionIndex = 0
        self.lowLevelPlan: List[Tuple[str, Tuple]] = []
        self.lowLevelActionIndex = 0
        self.startPosition = gameState.getAgentPosition(self.index)
        
        self._ghost_cache = []
        self._ghost_cache_state = None
        self._invader_cache = []
        self._invader_cache_state = None
        
        CaptureAgent.registerInitialState(self, gameState)
        
        self.walls = gameState.getWalls()
        W, H = self.walls.width, self.walls.height
        self.legalPositions = {
            (x, y) for x in range(W) for y in range(H) 
            if not self.walls[x][y]
        }
        
        self.isRed = gameState.isOnRedTeam(self.index)
        self.midX = W // 2
        homeX = self.midX - 1 if self.isRed else self.midX
        step = 1 if self.isRed else -1
        
        entries = []
        for y in range(H):
            a = (homeX, y)
            b = (homeX + step, y)
            if a in self.legalPositions and b in self.legalPositions:
                entries.append(a)
        
        if not entries:
            x_range = range(0, self.midX) if self.isRed else range(self.midX, W)
            entries = [(x, y) for (x, y) in self.legalPositions if x in x_range]
        
        self.homeEntries = entries
        self.homeScale = W + H
        
        # Map feature analysis
        self.map_width = W
        self.map_height = H
        self.map_type = self._analyzeMapType(gameState)
        
        self.training = False
        self.epsilon = 0.0
        self.alpha = 0.02
        self.discountRate = 0.9
        
        self.last_mode = "attack"
        self.stable_until = 0
        
        MixedAgent.CURRENT_ACTION[self.index] = {}
        
        if os.path.exists(MixedAgent.QLWeightsFile):
            try:
                with open(MixedAgent.QLWeightsFile, "r") as file:
                    MixedAgent.QLWeights = eval(file.read())
            except:
                pass

    def _analyzeMapType(self, gameState: GameState):
        W, H = self.walls.width, self.walls.height
        
        # Map feature analysis
        openness = len(self.legalPositions) / (W * H)
        aspect_ratio = W / H
        
        if openness > 0.6:
            return "open"
        elif openness < 0.4:
            return "cramped"
        else:
            return "normal"

    def final(self, gameState: GameState):
        if self.training:
            with open(MixedAgent.QLWeightsFile, 'w') as file:
                file.write(str(MixedAgent.QLWeights))

    def chooseAction(self, gameState: GameState):
        """Main decision loop - simplified to reduce replanning."""
        
        # Simplified mode decision (bypass PDDL for efficiency)
        mode = self._decideLowLevelMode(gameState)

        legalActions = gameState.getLegalActions(self.index)
        if not legalActions:
            return Directions.STOP

        bestAction = self._selectBestAction(gameState, mode, legalActions)
        return bestAction
    
    def _selectBestAction(self, gameState: GameState, mode: str, legalActions):
        if mode == "attack":
            featureFunc = self.getOffensiveFeatures
            weights = self.getOffensiveWeights()
        elif mode == "go_home":
            featureFunc = self.getEscapeFeatures
            weights = self.getEscapeWeights()
        else:
            featureFunc = self.getDefensiveFeatures
            weights = self.getDefensiveWeights()

        values = [(self.getQValue(featureFunc(gameState, a), weights), a) 
                for a in legalActions]
        return max(values)[1]

    def _decideLowLevelMode(self, gameState: GameState) -> str:
        """Adaptive decision-making - based on map and game situation"""
        my_state = gameState.getAgentState(self.index)
        my_pos = my_state.getPosition()
        
        ghosts = self._getGhostLocs(gameState)
        min_ghost_dist = min([self.getMazeDistance(my_pos, g) for g in ghosts]) if ghosts else 999
        
        invaders = self._getInvaders(gameState)
        
        # Dynamically calculate "safe carrying amount"
        total_food = len(self.getFood(gameState).asList())
        remaining_time = gameState.data.timeleft / 4  # Estimate remaining turns
        
        # Adjust carrying threshold based on remaining food and time
        if total_food <= 5:
            carry_threshold = 2  # Few food left, return quickly
        elif remaining_time < 100:
            carry_threshold = 3  # Time is short, score as soon as possible
        else:
            carry_threshold = 5  # Normal situation
        
        # Adjust danger distance based on map size
        danger_distance = max(3, self.homeScale // 10)  # Larger maps require longer safety distance

        if self.role == 'attacker':
            # Dynamically decide whether to return home
            if my_state.isPacman and min_ghost_dist <= danger_distance and my_state.numCarrying >= 2:
                return "go_home"
            
            if my_state.numCarrying >= carry_threshold:
                return "go_home"
            
            # Risk assessment
            risk_score = (carry_threshold - my_state.numCarrying) - (min_ghost_dist - danger_distance)
            if risk_score > 0:  # High risk
                return "go_home"
            
            return 'attack'

        else:  # defender
            if invaders:
                closest_inv_dist = min([self.getMazeDistance(my_pos, e.getPosition()) 
                                    for e in invaders])
                # Dynamic defense distance
                defend_distance = max(5, self.homeScale // 8)
                if closest_inv_dist <= defend_distance:
                    return 'defence'
            
            # Defender returns home after carrying a small amount
            if my_state.numCarrying >= max(2, carry_threshold - 2):
                return 'go_home'
            
            # Decide whether to assist in attack based on situation
            score = self.getScore(gameState)
            if score > 5 or (not invaders and min_ghost_dist > danger_distance * 2):
                return 'attack'
            
            return 'defence'


    def _getGhostLocs(self, gameState: GameState) -> List[Tuple[int, int]]:
        if not hasattr(self, '_ghost_cache') or self._ghost_cache_state != gameState:
            ghosts = []
            for opponent in self.getOpponents(gameState):
                opPos = gameState.getAgentPosition(opponent)
                opIsPacman = gameState.getAgentState(opponent).isPacman
                if opPos and not opIsPacman:
                    ghosts.append(opPos)
            self._ghost_cache = ghosts
            self._ghost_cache_state = gameState
        return self._ghost_cache

    def _getInvaders(self, gameState: GameState) -> List[AgentState]:
        if not hasattr(self, '_invader_cache') or self._invader_cache_state != gameState:
            enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            invaders = [e for e in enemies if e.isPacman and e.getPosition() is not None]
            self._invader_cache = invaders
            self._invader_cache_state = gameState
        return self._invader_cache

    def distanceToHome(self, gameState: GameState) -> int:
        myPos = gameState.getAgentPosition(self.index)
        if not myPos:
            return 0
        myPos = (int(myPos[0]), int(myPos[1]))
        valid_entries = [p for p in self.homeEntries if p in self.legalPositions]
        if not valid_entries:
            return 0
        try:
            dists = [self.getMazeDistance(myPos, p) for p in valid_entries]
            return min(dists)
        except:
            return 0

    def teamScore(self, gameState: GameState) -> int:
        return gameState.getScore() if gameState.isOnRedTeam(self.index) else -gameState.getScore()

    # === Q-LEARNING ===

    def getLowLevelPlanQL(self, gameState: GameState, mode: str) -> List[Tuple[str, Tuple]]:
        legalActions = gameState.getLegalActions(self.index)
        if not legalActions:
            return []
        
        if mode == "attack":
            rewardFunc = self.getOffensiveReward
            featureFunc = self.getOffensiveFeatures
            weights = self.getOffensiveWeights()
            learningRate = self.alpha
        elif mode == "go_home":
            rewardFunc = self.getEscapeReward
            featureFunc = self.getEscapeFeatures
            weights = self.getEscapeWeights()
            learningRate = self.alpha
        else:  # defence
            rewardFunc = self.getDefensiveReward
            featureFunc = self.getDefensiveFeatures
            weights = self.getDefensiveWeights()
            learningRate = self.alpha
        
        if self.training and util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            values = []
            for action in legalActions:
                if self.training:
                    self.updateWeights(gameState, action, rewardFunc, featureFunc, weights, learningRate)
                qval = self.getQValue(featureFunc(gameState, action), weights)
                values.append((qval, action))
            action = max(values)[1]
        
        myPos = gameState.getAgentPosition(self.index)
        nextPos = Actions.getSuccessor(myPos, action)
        return [(action, nextPos)]

    def posSatisfyLowLevelPlan(self, gameState: GameState) -> bool:
        if not self.lowLevelPlan or self.lowLevelActionIndex >= len(self.lowLevelPlan):
            return False
        myPos = gameState.getAgentPosition(self.index)
        action, target = self.lowLevelPlan[self.lowLevelActionIndex]
        nextPos = Actions.getSuccessor(myPos, action)
        return nextPos == target

    def getQValue(self, features, weights):
        return features * weights

    def updateWeights(self, gameState, action, rewardFunc, featureFunc, weights, learningRate):
        features = featureFunc(gameState, action)
        nextState = self.getSuccessor(gameState, action)
        reward = rewardFunc(gameState, nextState)
        
        for feature in features:
            correction = (reward + self.discountRate * self.getValue(nextState, featureFunc, weights)) - self.getQValue(features, weights)
            weights[feature] = weights[feature] + learningRate * correction * features[feature]

    def getValue(self, nextState: GameState, featureFunc, weights):
        legalActions = nextState.getLegalActions(self.index)
        if not legalActions:
            return 0.0
        qVals = [self.getQValue(featureFunc(nextState, a), weights) for a in legalActions]
        return max(qVals)

    # === FEATURES & REWARDS ===

    def getOffensiveFeatures(self, gameState: GameState, action):
        """Remove map-specific features"""
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        next_pos = successor.getAgentPosition(self.index)
        my_state = successor.getAgentState(self.index)
        
        features['bias'] = 1.0
        features['successorScore'] = self.getScore(successor)

        # Ghost check
        ghosts = self._getGhostLocs(gameState)
        if ghosts:
            min_ghost_dist = min(self.getMazeDistance(next_pos, g) for g in ghosts)
            features['#-of-ghosts-1-step-away'] = 1 if min_ghost_dist <= 1 else 0

        # Food distance - using normalization
        food_list = self.getFood(successor).asList()
        if food_list:
            manhattan_distances = [(abs(next_pos[0]-f[0]) + abs(next_pos[1]-f[1]), f) for f in food_list]
            manhattan_distances.sort()
            closest_foods = [f for _, f in manhattan_distances[:3]]
            min_food_dist = min(self.getMazeDistance(next_pos, food) for food in closest_foods)
            # Normalize to [0, 1]
            features['closest-food'] = min_food_dist / self.homeScale  # Divide by map size
        else:
            features['closest-food'] = 0

        # Return food value
        if my_state.numCarrying > 0:
            home_dist = self.distanceToHome(successor)
            # Normalization
            features['chance-return-food'] = (my_state.numCarrying / 10.0) / (home_dist / self.homeScale + 0.1)

        # Remove depth feature, or replace with normalized version
        # If keeping it, use relative depth:
        if my_state.isPacman:
            midX = self.midX
            if self.isRed:
                depth = max(0, next_pos[0] - midX)  
            else:
                depth = max(0, midX - next_pos[0])
            # Normalize to [0, 1]
            max_depth = (self.walls.width // 2)
            features['depth'] = depth / max(max_depth, 1)  # Relative depth

        features['stop'] = 1 if action == Directions.STOP else 0
        prev_action = gameState.getAgentState(self.index).configuration
        if prev_action:
            features['reverse'] = 1 if action == Directions.REVERSE[prev_action.direction] else 0
        
        return features


    def getOffensiveReward(self, gameState: GameState, nextState: GameState):
        currState = gameState.getAgentState(self.index)
        nextAgentState = nextState.getAgentState(self.index)
        
        reward = -1

        new_returned = nextAgentState.numReturned - currState.numReturned
        if new_returned > 0:
            reward += new_returned * 50  

        if nextAgentState.numCarrying > currState.numCarrying:
            reward += 10 

        if currState.numCarrying > 0 and nextAgentState.numCarrying == 0 and nextAgentState.numReturned == currState.numReturned:
            reward -= currState.numCarrying * 30  

        ghosts = self._getGhostLocs(nextState)
        if ghosts:
            myPos = nextAgentState.getPosition()
            #min_ghost_dist = min([self.getMazeDistance(myPos, g) for g in ghosts])
            min_ghost_dist = min(self.getMazeDistance(myPos, g) for g in ghosts)
            if min_ghost_dist <= 1:
                reward -= 30  

        reward += nextAgentState.numCarrying * 2
        
        return reward

    def getOffensiveWeights(self):
        return MixedAgent.QLWeights["offensiveWeights"]

    def getDefensiveFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        
        features['onDefense'] = 0 if myState.isPacman else 1
        
        invaders = self._getInvaders(successor)
        features['numInvaders'] = len(invaders)
        
        if invaders:
            dists = [self.getMazeDistance(myPos, e.getPosition()) for e in invaders]
            features['invaderDistance'] = min(dists)
        else:
            features['invaderDistance'] = 0
        
        teammates = [i for i in self.getTeam(successor) if i != self.index]
        if teammates:
            teammate_pos = successor.getAgentPosition(teammates[0])
            if teammate_pos:
                features['teamDistance'] = self.getMazeDistance(myPos, teammate_pos)
        
        defendingFood = self.getFoodYouAreDefending(successor).asList()
        if defendingFood:
            min_food_dist = min([self.getMazeDistance(myPos, food) for food in defendingFood])
            features['distanceToFood'] = min_food_dist
        
        features['blockingEscape'] = 0
        if invaders and self.homeEntries:
            for invader in invaders:
                inv_pos = invader.getPosition()

                inv_home_dists = [self.getMazeDistance(inv_pos, entry) for entry in self.homeEntries]
                my_home_dists = [self.getMazeDistance(myPos, entry) for entry in self.homeEntries]
                if min(my_home_dists) < min(inv_home_dists):
                    features['blockingEscape'] = 1
                    break
    
        currConf = gameState.getAgentState(self.index).configuration
        if currConf:
            rev = Directions.REVERSE[currConf.direction]
            features['reverse'] = 1 if action == rev else 0
        features['stop'] = 1 if action == Directions.STOP else 0
        
        return features

    def getDefensiveReward(self, gameState, nextState):
        reward = -1
        
        if not nextState.getAgentState(self.index).isPacman:
            reward += 5
        
        curr_invaders = len(self._getInvaders(gameState))
        next_invaders = len(self._getInvaders(nextState))
        if next_invaders < curr_invaders:
            reward += 100
        
        curr_food = len(self.getFoodYouAreDefending(gameState).asList())
        next_food = len(self.getFoodYouAreDefending(nextState).asList())
        if next_food < curr_food:
            reward -= 50
        
        return reward

    def getDefensiveWeights(self):
        return MixedAgent.QLWeights["defensiveWeights"]

    def getEscapeFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        walls = gameState.getWalls()
        board_scale = float(walls.width + walls.height)
        
        home_dist = self.distanceToHome(successor)
        features['distanceToHome'] = home_dist / board_scale
        
        ghosts = self._getGhostLocs(successor)
        if ghosts:
            #min_ghost_dist = min([self.getMazeDistance(myPos, g) for g in ghosts])
            min_ghost_dist = min(self.getMazeDistance(myPos, g) for g in ghosts)
            features['enemyDistance'] = min_ghost_dist
            features['veryCloseGhost'] = 1 if min_ghost_dist <= 2 else 0
        else:
            features['enemyDistance'] = board_scale
            features['veryCloseGhost'] = 0
        
        features['carryingFood'] = myState.numCarrying / 10.0
        features['onDefense'] = 0 if myState.isPacman else 1
        
        currConf = gameState.getAgentState(self.index).configuration
        if currConf:
            rev = Directions.REVERSE[currConf.direction]
            features['reverse'] = 1 if action == rev else 0
        features['stop'] = 1 if action == Directions.STOP else 0
        
        return features

    def getEscapeReward(self, gameState, nextState):
        currState = gameState.getAgentState(self.index)
        nextAgentState = nextState.getAgentState(self.index)
        
        reward = -1
        
        if currState.isPacman and not nextAgentState.isPacman:
            reward += nextAgentState.numReturned * 30
        
        if currState.numCarrying > 0 and nextAgentState.numCarrying == 0 and nextAgentState.numReturned == currState.numReturned:
            reward -= currState.numCarrying * 50
        
        curr_home_dist = self.distanceToHome(gameState)
        next_home_dist = self.distanceToHome(nextState)
        if next_home_dist < curr_home_dist:
            reward += 3
        
        curr_ghosts = self._getGhostLocs(gameState)
        next_ghosts = self._getGhostLocs(nextState)
        if curr_ghosts and next_ghosts:
            curr_pos = currState.getPosition()
            next_pos = nextAgentState.getPosition()
            curr_min = min([self.getMazeDistance(curr_pos, g) for g in curr_ghosts])
            next_min = min([self.getMazeDistance(next_pos, g) for g in next_ghosts])
            if next_min > curr_min:
                reward += 5
        
        return reward

    def getEscapeWeights(self):
        return MixedAgent.QLWeights["escapeWeights"]

    # === UTILITY ===

    def closestFood(self, pos, food, walls):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            if food[pos_x][pos_y]:
                return dist
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist + 1))
        return None

    def getSuccessor(self, gameState: GameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor