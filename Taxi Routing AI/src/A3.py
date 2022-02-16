import matplotlib.pyplot as plt
import numpy as np
import random
import time
import sys

# initial seed for reproducibility
random.seed(49)

# squareGrid to handle Taxi Domain instances
class squareGrid:
    # initialize grid
    def __init__(self, size, depotList, walls, destination, start=(-1, -1), source=(-1, -1)):
        # attribute names explain their purpose. refer report for definitions
        self.size = size
        self.depotList = depotList
        self.walls = walls
        self.destination = destination
        self.terminalState = (destination[0], destination[1], destination[0], destination[1], 0)
        self.actions = ['N', 'E', 'S', 'W', 'U', 'D']
        self.actionToWord = {
            'N': "North",
            'E': "East",
            'S': "South",
            'W': "West",
            'U': "Pickup",
            'D': "Putdown"
        }
        self.start = start
        self.source = source
        # used with decaying exploration
        self.decay_factor = 1
        self.stateSpace = []
        self.state2idx = {}
        # used in value-iteration
        self.valueBefore = {}
        self.valueAfter = {}
        # used in policy-iteration
        self.policy = {}
        self.optimalPolicy = {}
        # used in RL-algorithms
        self.qValue = {}
        # generation of state space according to given constraints (refer report)
        for taxi_x in range(size):
            for taxi_y in range(size):
                for pass_x in range(size):
                    for pass_y in range(size):
                        for isPicked in range(2):
                            if isPicked == 1 and (taxi_x, taxi_y) != (pass_x, pass_y):
                                continue
                            if (pass_x, pass_y) == destination and (taxi_x, taxi_y) != destination:
                                continue
                            self.stateSpace.append((taxi_x, taxi_y, pass_x, pass_y, isPicked))
                            self.valueBefore[(taxi_x, taxi_y, pass_x, pass_y, isPicked)] = 0
                            self.valueAfter[(taxi_x, taxi_y, pass_x, pass_y, isPicked)] = 0
                            self.policy[(taxi_x, taxi_y, pass_x, pass_y, isPicked)] = 'NA'
                            self.optimalPolicy[(taxi_x, taxi_y, pass_x, pass_y, isPicked)] = 'NA'
        idx = 0
        for s in self.stateSpace:
            for a in self.actions:
                self.qValue[(s, a)] = 0
            if s != self.terminalState:
                self.state2idx[s] = idx
                idx += 1
        self.state2idx[self.terminalState] = idx
        self.idx2state = {}
        for k, v in self.state2idx.items():
            self.idx2state[v] = k
    # initialize instance by specifying start and source locations. Reset values and policies
    def initInstance(self, start=(-1, -1), source=(-1, -1)):
        self.start = start
        self.source = source
        for state in self.stateSpace:
            self.valueBefore[state] = 0
            self.valueAfter[state] = 0
            if state == self.terminalState:
                self.policy[state] = 'NA'
                self.optimalPolicy[state] = 'NA'
            else:
                self.policy[state] = random.choice(self.actions)
                self.optimalPolicy[state] = random.choice(self.actions)
            for action in self.actions:
                self.qValue[(state, action)] = 0
    # get next state (use transition model for stochasticity)
    def nextState(self, s, a):
        # state is a 5-tuple
        # t: taxi coordinates
        # p: passenger coordinates
        # b:  picked or not (boolean)
        t_x, t_y, p_x, p_y, b = s
        if a == 'U':
            if (t_x, t_y) == (p_x, p_y):
                return (t_x, t_y, p_x, p_y, 1)
            else:
                return s
        elif a == 'D':
            if (t_x, t_y) == (p_x, p_y):
                return (t_x, t_y, p_x, p_y, 0)
            else:
                return s
        elif a in ['N', 'E', 'S', 'W']:
            coord_N = (t_x, min(t_y + 1, self.size - 1))
            coord_E = (min(t_x + 1, self.size - 1), t_y)
            coord_S = (t_x, max(t_y - 1, 0))
            coord_W = (max(t_x - 1, 0), t_y)
            if ((t_x, t_y), coord_N) in self.walls or (coord_N, (t_x, t_y)) in self.walls:
                coord_N = (t_x, t_y)
            if ((t_x, t_y), coord_E) in self.walls or (coord_E, (t_x, t_y)) in self.walls:
                coord_E = (t_x, t_y)
            if ((t_x, t_y), coord_S) in self.walls or (coord_S, (t_x, t_y)) in self.walls:
                coord_S = (t_x, t_y)
            if ((t_x, t_y), coord_W) in self.walls or (coord_W, (t_x, t_y)) in self.walls:
                coord_W = (t_x, t_y)
            prob_N = (0.85 if a == 'N' else 0.05)
            prob_E = (0.85 if a == 'E' else 0.05)
            prob_S = (0.85 if a == 'S' else 0.05)
            sample = random.uniform(0, 1)
            if sample <= prob_N:
                t_x_, t_y_ = coord_N
            elif sample <= prob_N + prob_E:
                t_x_, t_y_ = coord_E
            elif sample <= prob_N + prob_E + prob_S:
                t_x_, t_y_ = coord_S
            else:
                t_x_, t_y_ = coord_W
            if b == 1:
                return (t_x_, t_y_, t_x_, t_y_, 1)
            else:
                return (t_x_, t_y_, p_x, p_y, 0)
    # probability function: transition = s * a -> s' (refer report for transition model specs)
    def probability(self, s, a, s_):
        # state is a 5-tuple
        # t: taxi coordinates
        # p: passenger coordinates
        # b:  picked or not (boolean)
        t_x, t_y, p_x, p_y, b = s
        t_x_, t_y_, p_x_, p_y_, b_ = s_
        if a == 'U':
            if (t_x, t_y) == (t_x_, t_y_) and (p_x, p_y) == (p_x_, p_y_):
                if (t_x, t_y) == (p_x, p_y):
                    return b_
                else:
                    return 1
            else:
                return 0
        elif a == 'D':
            if (t_x, t_y) == (t_x_, t_y_) and (p_x, p_y) == (p_x_, p_y_):
                if (t_x, t_y) == (p_x, p_y):
                    return not b_
                else:
                    return 1
            else:
                return 0
        elif a in ['N', 'E', 'S', 'W']:
            if b != b_:
                return 0
            if b == 0 and (p_x, p_y) != (p_x_, p_y_):
                return 0
            coord_N = (t_x, min(t_y + 1, self.size - 1))
            coord_E = (min(t_x + 1, self.size - 1), t_y)
            coord_S = (t_x, max(t_y - 1, 0))
            coord_W = (max(t_x - 1, 0), t_y)
            if ((t_x, t_y), coord_N) in self.walls or (coord_N, (t_x, t_y)) in self.walls:
                coord_N = (t_x, t_y)
            if ((t_x, t_y), coord_E) in self.walls or (coord_E, (t_x, t_y)) in self.walls:
                coord_E = (t_x, t_y)
            if ((t_x, t_y), coord_S) in self.walls or (coord_S, (t_x, t_y)) in self.walls:
                coord_S = (t_x, t_y)
            if ((t_x, t_y), coord_W) in self.walls or (coord_W, (t_x, t_y)) in self.walls:
                coord_W = (t_x, t_y)
            prob = 0
            if (t_x_, t_y_) == coord_N:
                prob += (0.85 if a == 'N' else 0.05)
            if (t_x_, t_y_) == coord_E:
                prob += (0.85 if a == 'E' else 0.05)
            if (t_x_, t_y_) == coord_S:
                prob += (0.85 if a == 'S' else 0.05)
            if (t_x_, t_y_) == coord_W:
                prob += (0.85 if a == 'W' else 0.05)
            return prob
    # reward function: transition = s * a -> s' (refer report for reward model specs)
    def reward(self, s, a, s_):
        # check for valid transition
        if self.probability(s, a, s_) == 0:
            return 0
        # state is a 5-tuple
        # t: taxi coordinates
        # p: passenger coordinates
        t_x, t_y, p_x, p_y, _ = s
        if a in ['N', 'E', 'S', 'W']:
            return -1
        elif a == 'U':
            if (t_x, t_y) != (p_x, p_y):
                return -10
            else:
                return -1
        else:
            if (t_x, t_y) != (p_x, p_y):
                return -10
            else:
                if s_ == self.terminalState:
                    return 20
                else:
                    return -1
    # value iteration algorithm
    def valueIteration(self, gamma, epsilon, verbose=True):
        # gamma: discount factor
        # epsilon: max-norm threshold
        iterationIndexList = []
        valueList = []
        iterationIndex = 0
        # repeat till convergence
        while True:
            iterationIndex += 1
            maxBellmanUpdate = -float('inf')
            # standard Bellman Value Iteration
            for s in self.stateSpace:
                if s == self.terminalState:
                    continue
                maxValue = -float('inf')
                for a in self.actions:
                    value = 0
                    for s_ in self.stateSpace:
                        prob = self.probability(s, a, s_)
                        if prob != 0:
                            value += prob * (self.reward(s, a, s_) + gamma * self.valueBefore[s_])
                    maxValue = max(maxValue, value)
                self.valueAfter[s] = maxValue
                maxBellmanUpdate = max(maxBellmanUpdate, abs(self.valueAfter[s] - self.valueBefore[s]))
            iterationIndexList.append(iterationIndex)
            valueList.append([self.valueAfter[s] for s in self.stateSpace])
            for s in self.stateSpace:
                self.valueBefore[s] = self.valueAfter[s]
                self.valueAfter[s] = 0
            if verbose:
                print("Iteration: " + str(iterationIndex) + ", Max. Bellman Update: " + str(maxBellmanUpdate))
            # convergence criteria
            if maxBellmanUpdate * gamma < epsilon * (1 - gamma):
                break
        # algorithm converged
        maxNormDistanceList = []
        last = len(valueList) - 1
        for i in range(last + 1):
            maxNormDistance = 0
            for j in range(len(self.stateSpace)):
                maxNormDistance = max(maxNormDistance, abs(valueList[i][j] - valueList[last][j]))
            maxNormDistanceList.append(maxNormDistance)
        # return list
        return iterationIndexList, maxNormDistanceList            
    # policy iteration algorithm
    def policyIteration(self, gamma, epsilon, analytical=False, verbose=True):
        # gamma: discount factor
        # epsilon: max-norm threshold
        iterationIndexList = []
        valueList = []
        # assign random policy
        for state in self.stateSpace:
            if state == self.terminalState:
                continue
            self.policy[state] = random.choice(self.actions)
        iterationIndex = 0
        # policy evaluation
        while True:
            iterationIndex += 1
            # policy evaluation
            # refer report for analytical solution expression
            if analytical:
                size = self.state2idx[self.terminalState] + 1
                V = np.zeros((size, 1))
                P = np.zeros((size, size))
                R = np.zeros((size, size))
                for i in range(size):
                    if i == size - 1:
                        P[i][i] = 1
                        continue
                    for j in range(size):
                        P[i][j] = self.probability(self.idx2state[i], self.policy[self.idx2state[i]], self.idx2state[j])
                        R[i][j] = self.reward(self.idx2state[i], self.policy[self.idx2state[i]], self.idx2state[j])
                I = np.eye(size)
                B = np.diagonal(P @ R.T).reshape((size, 1))
                # (I - gamma * P)V = B
                V = np.linalg.inv(I - gamma * P) @ B
                for s in self.stateSpace:
                    if s == self.terminalState:
                        continue
                    self.valueBefore[s] = V[self.state2idx[s]][0]
            # iterative method (modified bellman update)
            else:
                iter = 0
                # repeat till convergence
                while True:
                    iter += 1
                    maxBellmanUpdate = -float('inf')
                    for s in self.stateSpace:
                        if s == self.terminalState:
                            continue
                        value = 0
                        a = self.policy[s]
                        for s_ in self.stateSpace:
                            prob = self.probability(s, a, s_)
                            if prob != 0:
                                value += prob * (self.reward(s, a, s_) + gamma * self.valueBefore[s_])
                        self.valueAfter[s] = value
                        maxBellmanUpdate = max(maxBellmanUpdate, abs(self.valueAfter[s] - self.valueBefore[s]))
                    for s in self.stateSpace:
                        self.valueBefore[s] = self.valueAfter[s]
                        self.valueAfter[s] = 0
                    # convergence criteria
                    if maxBellmanUpdate * gamma < epsilon * (1 - gamma):
                        break
            valueList.append([self.valueBefore[s] for s in self.stateSpace])
            iterationIndexList.append(iterationIndex)
            # policy improvement (policy-update)
            unchanged = True
            for s in self.stateSpace:
                if s == self.terminalState:
                    continue
                maxValue = 0
                for s_ in self.stateSpace:
                    prob = self.probability(s, self.policy[s], s_)
                    if prob != 0:
                        maxValue += prob * (self.reward(s, self.policy[s], s_) + gamma * self.valueBefore[s_])
                maxAction = self.policy[s]
                for a in self.actions:
                    value = 0
                    for s_ in self.stateSpace:
                        prob = self.probability(s, a, s_)
                        if prob != 0:
                            value += prob * (self.reward(s, a, s_) + gamma * self.valueBefore[s_])
                    # 1e-6 provides stability and prevents oscillations
                    if value > maxValue + 1e-6:
                        maxValue = value
                        maxAction = a
                if self.policy[s] != maxAction:
                    unchanged = False
                    self.policy[s] = maxAction
            # convergence criteria
            if unchanged:
                break
            if verbose:
                print("Policy Iteration: " + str(iterationIndex) + ", Policy improved")
        # algorithm converged
        policyLossList = []
        last = len(valueList) - 1
        for i in range(last + 1):
            maxPolicyLoss = 0
            for j in range(len(self.stateSpace)):
                maxPolicyLoss = max(maxPolicyLoss, abs(valueList[i][j] - valueList[last][j]))
            policyLossList.append(maxPolicyLoss)
        # return list
        return iterationIndexList, policyLossList
    # determine optimal policy from computed values (state)
    def computePolicy(self, gamma, useQ=False):
        # gamma: discount factor
        # useQ: whether to use Q-Value or Value for policy determination
        # boolean to check if optimal policy changed
        unchanged = True
        for s in self.stateSpace:
            if s == self.terminalState:
                continue
            if useQ:
                maxAction = self.optimalPolicy[s]
                maxValue = self.qValue[(s, maxAction)]
            else:
                maxAction = 'NA'
                maxValue = -float('inf')
            for a in self.actions:
                if useQ:
                    value = self.qValue[(s, a)]
                else:
                    value = 0
                    for s_ in self.stateSpace:
                        prob = self.probability(s, a, s_)
                        if prob != 0:
                            value += prob * (self.reward(s, a, s_) + gamma * self.valueBefore[s_])
                # 1e-3 provides stability and prevents oscillations
                if value > maxValue + 1e-3:
                    maxValue = value
                    maxAction = a
            if self.optimalPolicy[s] != maxAction:
                unchanged = False
            self.optimalPolicy[s] = maxAction
        return unchanged
    # Q-learning with epsilon-greedy approach
    def qLearning(self, gamma, alpha, epsilon, exploration='fixed', max_iter=500):
        discountedReward = 0
        iterationIndex = 0
        currState = (self.start[0], self.start[1], self.source[0], self.source[1], 0)
        while iterationIndex < max_iter:
            sample = random.uniform(0, 1)
            if sample < (epsilon / self.decay_factor):
                # explore randomly
                action = random.choice(self.actions)
            else:
                # choose best action
                action = 'N'
                qValue = self.qValue[(currState, action)]
                for a in self.actions:
                    if self.qValue[(currState, a)] > qValue:
                        action = a
                        qValue = self.qValue[(currState, a)]
            # get next state and reward
            nextState = self.nextState(currState, action)
            reward = self.reward(currState, action, nextState)
            # update accumulated reward
            discountedReward += (gamma ** (iterationIndex)) * reward
            # greedy one-step look ahead
            maxQ = -float('inf')
            if nextState == self.terminalState:
                maxQ = 0
            else:
                for a in self.actions:
                    if self.qValue[(nextState, a)] > maxQ:
                        maxQ = self.qValue[(nextState, a)]
            # update Q-value
            self.qValue[(currState, action)] += alpha * (reward + gamma * maxQ - self.qValue[(currState, action)])
            # update state
            currState = nextState
            if currState == self.terminalState:
                break
            # update decay factor and iteration index
            if exploration != 'fixed':
                self.decay_factor += 1
            iterationIndex += 1
        # return discounted reward
        return discountedReward
    # Q-learning with SARSA
    def SARSA(self, gamma, alpha, epsilon, exploration='fixed', max_iter=500):
        discountedReward = 0
        iterationIndex = 0
        currState = (self.start[0], self.start[1], self.source[0], self.source[1], 0)
        action = 'NA'
        while iterationIndex < max_iter:
            if iterationIndex == 0:
                sample = random.uniform(0, 1)
                if sample < (epsilon / self.decay_factor):
                    # explore randomly
                    action = random.choice(self.actions)
                else:
                    # choose best action
                    action = 'N'
                    qValue = self.qValue[(currState, action)]
                    for a in self.actions:
                        if self.qValue[(currState, a)] > qValue:
                            action = a
                            qValue = self.qValue[(currState, a)]
                if exploration != 'fixed':
                    self.decay_factor += 1
            # get next state and reward
            nextState = self.nextState(currState, action)
            reward = self.reward(currState, action, nextState)
            # update accumulated reward
            discountedReward += (gamma ** (iterationIndex)) * reward
            if nextState == self.terminalState:
                # update Q-value
                self.qValue[(currState, action)] += alpha * (reward - self.qValue[(currState, action)])
            else:
                # choose action for next state
                sample = random.uniform(0, 1)
                if sample < (epsilon / self.decay_factor):
                    # explore randomly
                    a = random.choice(self.actions)
                else:
                    # choose best action
                    a = 'N'
                    qValue = self.qValue[(nextState, a)]
                    for a_ in self.actions:
                        if self.qValue[(nextState, a_)] > qValue:
                            a = a_
                            qValue = self.qValue[(nextState, a_)]
                # update Q-value
                self.qValue[(currState, action)] += alpha * (reward + gamma * self.qValue[(nextState, a)] - self.qValue[(currState, action)])
                # update action
                action = a
            # update state
            currState = nextState
            if currState == self.terminalState:
                break
            # update decay factor and iteration index
            if exploration != 'fixed':
                self.decay_factor += 1
            iterationIndex += 1
        # return discounted reward
        return discountedReward

# driver for A1, simulate policy
# gridWorld: problem instance
# max_iter: maximum state updates
# gamma: discount factor
# verbose: output verbosity
def A1(gridWorld: squareGrid, max_iter=500, gamma=1, verbose=True):
    # get the taxi, passenger source and passenger destination
    start = gridWorld.start
    source = gridWorld.source
    destination = gridWorld.destination
    # define current state tuple
    currState = (start[0], start[1], source[0], source[1], 0)
    if verbose:
        print("Taxi starting at location: " + str(start))
        print("Passenger (source) at location: " + str(source))
        print("Passenger (destination) at location: " + str(destination))
        print("---------------------------------------------------------")
        print("Starting simulation... (Max. updates = " + str(max_iter) + ")")
    # store iteration number and sum of discounted rewards
    iterationIndex = 0
    discountedReward = 0
    # flag for successful termination (destination reached)
    finish = False
    # iterate for a total of max_iter times
    while iterationIndex < max_iter:
        iterationIndex += 1
        # take action according to optimal policy
        action = gridWorld.optimalPolicy[currState]
        # get next state from the environment
        nextState = gridWorld.nextState(currState, action)
        # get suitable reward
        reward = gridWorld.reward(currState, action, nextState)
        # accumulate sum of discounted rewards
        discountedReward += (gamma ** (iterationIndex - 1)) * reward
        if verbose:
            print("Update " + str(iterationIndex) + ": " + str(currState) + " * " + str(gridWorld.actionToWord[action]) + " -> " + str(nextState))
        # update agent's state
        currState = nextState
        # check for goal state (terminal state)
        if currState == gridWorld.terminalState:
            finish = True
            break
    if verbose:
        # print termination message based on 'finish'
        if finish:
            print("Stopping simulation... Destination reached.")
        else:
            print("Stopping simulation... Max. updates done.")
        print("---------------------------------------------------------")
        # print sum of discounted rewards
        print("Discounted Reward: " + str(discountedReward))
    # return reward
    return discountedReward 

# driver for A2, implement value iteration
# part_num: part number
# start: taxi start location
# source: passenger source location
# destination: passenger destination location
# verbose: output verbosity
# epsilon: convergence threshold
def A2(part_num, start=(3, 0), source=(0, 0), destination=(4, 4), verbose=True, epsilon=0.01):
    # define grid size, depot and wall locations
    size = 5
    depotList = [(0, 0), (0, 4), (3, 0), (4, 4)]
    walls = [((0, 0), (1, 0)), ((0, 1), (1, 1)), ((2, 0), (3, 0)), ((2, 1), (3, 1)), ((1, 3), (2, 3)), ((1, 4), (2, 4))]
    if part_num == 'a':
        # part (a), initialize instance (start, source and destination), create grid-world
        gridWorld = squareGrid(size, depotList, walls, destination, start, source)
        # set discount factor
        gamma = 0.9
        # initialize values and policies
        gridWorld.initInstance(start, source)
        # implement value iteration
        iterationIndexList, _ = gridWorld.valueIteration(gamma, epsilon, verbose)
        print("Number of iterations: " + str(len(iterationIndexList)))
        # simulate optimal policy
        gridWorld.computePolicy(gamma)
        A1(gridWorld, 50, gamma)
    elif part_num == 'b':
        # part (b), initialize instance (start, source and destination), create grid-world
        gridWorld = squareGrid(size, depotList, walls, destination, start, source)
        # list of discount factors to experiment with
        discountFactors = [0.01, 0.1, 0.5, 0.8, 0.99]
        print("Taxi Start:", start)
        print("Passenger Start:", source)
        print("Passenger Destination:", destination)
        for i in range(len(discountFactors)):
            # get gamma
            gamma = discountFactors[i]
            # initialize values and policies
            gridWorld.initInstance(start, source)
            # perform value iteration
            iterationIndexList, maxNormDistanceList = gridWorld.valueIteration(gamma, epsilon, verbose)
            print("Gamma:", gamma)
            print("Number of Iterations:", len(iterationIndexList))
            # plot corresponding graph
            plt.figure(i)
            plt.plot(iterationIndexList, maxNormDistanceList)
            plt.xlabel("Number of iterations")
            plt.ylabel("Max. Norm Distance")
            plt.title("Value iteration ($\gamma =$ " + str(gamma) + ")")
            plt.savefig("A2_b_" + str(i) + ".png")
    else:
        # part (c), initialize instance (learn policy for gamma = 0.1 and 0.99)
        sourceDest = random.sample(depotList, 2)
        # fix goal state (given in assigment)
        sourceDest[1] = (4, 4)
        # random taxi start state
        start = (random.choice([i for i in range(size)]), random.choice([i for i in range(size)]))
        # create 2 grid-worlds (one for gamma = 0.1 and other for gamma = 0.99)
        gridWorld = [squareGrid(size, depotList, walls, sourceDest[1], start, sourceDest[0]), squareGrid(size, depotList, walls, sourceDest[1], start, sourceDest[0])]
        # define list of discount factors
        discountFactors = [0.1, 0.99]
        for i in range(len(discountFactors)):
            gamma = discountFactors[i]
            # initialize values and policies
            gridWorld[i].initInstance(start, sourceDest[0])
            # perform value iteration
            gridWorld[i].valueIteration(gamma, epsilon, verbose)
            # compute optimal policy
            gridWorld[i].computePolicy(gamma)
        # list of (start, source) locations to experiment with (all depot combinations)
        source = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 4), (0, 4), (0, 4), (0, 4), (3, 0), (3, 0), (3, 0), (3, 0)]
        start = [(0, 4), (0, 0), (3, 0), (4, 4), (0, 0), (0, 4), (3, 0), (4, 4), (0, 0), (0, 4), (3, 0), (4, 4)]
        destination = (4, 4)
        # execute optimal policy on all start state configurations for both gamma values
        for i in range(12):
            for j in range(2):
                gamma = discountFactors[j]
                # set start, source and destination
                gridWorld[j].start = start[i]
                gridWorld[j].source = source[i]
                gridWorld[j].destination = destination
                print("Gamma:", gamma)
                # execute policy (20 steps only)
                A1(gridWorld[j], 20, gamma)
                print("----------------------------------")
            print("_______________________________________")

# driver for A3, implement policy iteration
# verbose: output verbosity
def A3(verbose=True):
    # define grid size, depot and wall locations
    size = 5
    depotList = [(0, 0), (0, 4), (3, 0), (4, 4)]
    walls = [((0, 0), (1, 0)), ((0, 1), (1, 1)), ((2, 0), (3, 0)), ((2, 1), (3, 1)), ((1, 3), (2, 3)), ((1, 4), (2, 4))]
    # initialize example instance (random start, source and destination)
    sourceDest = random.sample(depotList, 2)
    start = (random.choice([i for i in range(size)]), random.choice([i for i in range(size)]))
    # create grid world
    gridWorld = squareGrid(size, depotList, walls, sourceDest[1], start, sourceDest[0])
    # list of discount factors to experiment with
    discountFactors = [0.01, 0.1, 0.5, 0.8, 0.99]
    # convergence threshold (policy evaluation)
    epsilon = 0.01
    # list to store time taken by iterative and analytical method respectively
    iterative_time = []
    analytical_time = []
    for i in range(len(discountFactors)):
        # set discount factor
        gamma = discountFactors[i]
        # determine optimal policy using iterative method
        # initialize values and policy
        gridWorld.initInstance(start, sourceDest[0])
        # note time taken
        iterative_start = time.time()
        iterationIndexList, policyLossList = gridWorld.policyIteration(gamma, epsilon, False, verbose)
        iterative_end = time.time()
        # determine optimal policy using analytical method
        # initialize values and policy
        gridWorld.initInstance(start, sourceDest[0])
        # note time taken
        analytical_start = time.time()
        iterationIndexList, policyLossList = gridWorld.policyIteration(gamma, epsilon, True, verbose)
        analytical_end = time.time()
        print("Gamma:", gamma)
        # print time taken by two approaches
        print("Time taken (iterative, in seconds):", iterative_end - iterative_start)
        print("Time taken (linalg, in seconds)", analytical_end - analytical_start)
        # append time taken into corresponding lists
        iterative_time.append(iterative_end - iterative_start)
        analytical_time.append(analytical_end - analytical_start)
        # plot policy loss vs iteration number
        plt.figure(i + 1)
        plt.plot(iterationIndexList, policyLossList)
        plt.xlabel("Number of iterations")
        plt.ylabel("Policy Loss")
        plt.title("Policy iteration ($\gamma =$ " + str(gamma) + ")")
        plt.savefig("A3_" + str(i) + ".png")
    # plot graph of computation/learning time vs discount factor (both iterative and analytical)
    plt.figure(0)
    plt.plot(discountFactors, iterative_time, label='Iterative')
    plt.plot(discountFactors, analytical_time, label='Analytical')
    plt.xlabel("Discount Factor")
    plt.ylabel("Learning Time (in seconds)")
    plt.title("Learning Time vs Gamma")
    plt.legend()
    plt.savefig("A3_time.png")
    # testing optimal policy
    gridWorld.computePolicy(0.99)
    A1(gridWorld, 50, 0.99)

# max_episodes: maximum number of episodes to train RL algorithm for
# plot: boolean (whether or not to plot graphs)
# avg_over: number of episodes to use for a particular start state
def B2(max_episodes, plot=True, avg_over=80):
    # set RL parameters (learning rate, discount factor, exploration rate)
    alpha = 0.25
    gamma = 0.99
    epsilon = 0.1
    # set grid size, depot and wall locations
    size = 5
    depotList = [(0, 0), (0, 4), (3, 0), (4, 4)]
    walls = [((0, 0), (1, 0)), ((0, 1), (1, 1)), ((2, 0), (3, 0)), ((2, 1), (3, 1)), ((1, 3), (2, 3)), ((1, 4), (2, 4))]
    # select destination (randomly)
    sourceDest = random.sample(depotList, 2)
    # create grid worlds for all 4 RL algorithms
    a_gridWorld = squareGrid(size, depotList, walls, sourceDest[1])
    b_gridWorld = squareGrid(size, depotList, walls, sourceDest[1])
    c_gridWorld = squareGrid(size, depotList, walls, sourceDest[1])
    d_gridWorld = squareGrid(size, depotList, walls, sourceDest[1])
    # initialise values and policies
    a_gridWorld.initInstance()
    b_gridWorld.initInstance()
    c_gridWorld.initInstance()
    d_gridWorld.initInstance()
    # possible values for source
    sourceDepotList = [depot for depot in depotList if depot != sourceDest[1]]
    # reward lists for book-keeping
    a_rewardList = []
    b_rewardList = []
    c_rewardList = []
    d_rewardList = []
    # form list of possible start states (75 such states)
    start_state_list = []
    for depot in sourceDepotList:
        for i in range(size):
            for j in range(size):
                start_state_list.append((i, j, depot[0], depot[1], 0))
    # run algorithms for max episodes
    for t in range(max_episodes):
        # Q-Learning (fixed exploration)
        # choose random start state
        start = (random.choice([i for i in range(size)]), random.choice([i for i in range(size)]))
        source = random.choice(sourceDepotList)
        a_gridWorld.start = start
        a_gridWorld.source = source
        # learn and update optimal policy
        a_gridWorld.qLearning(gamma, alpha, epsilon)
        a_gridWorld.computePolicy(gamma, True)
        # test optimal policy (every 20 episodes)
        if (t + 1) % 20 == 0:
            expected_utility = 0
            for _ in range(avg_over):
                for taxi_x, taxi_y, pass_x, pass_y, _ in start_state_list:
                    # random start state
                    a_gridWorld.start = (taxi_x, taxi_y)
                    a_gridWorld.source = (pass_x, pass_y)
                    # get utility value
                    expected_utility += A1(a_gridWorld, 50, gamma, False)
            expected_utility /= (avg_over * len(start_state_list))
            # append in reward list
            a_rewardList.append(expected_utility)
        # Q-Learning (decaying exploration)
        # choose random start state
        start = (random.choice([i for i in range(size)]), random.choice([i for i in range(size)]))
        source = random.choice(sourceDepotList)
        b_gridWorld.start = start
        b_gridWorld.source = source
        # learn and update optimal policy
        b_gridWorld.qLearning(gamma, alpha, epsilon, 'decay')
        b_gridWorld.computePolicy(gamma, True)
        # test optimal policy (every 20 episodes)
        if (t + 1) % 20 == 0:    
            expected_utility = 0
            for _ in range(avg_over):
                for taxi_x, taxi_y, pass_x, pass_y, _ in start_state_list:
                    # random start state
                    b_gridWorld.start = (taxi_x, taxi_y)
                    b_gridWorld.source = (pass_x, pass_y)
                    # get utility value
                    expected_utility += A1(b_gridWorld, 50, gamma, False)
            expected_utility /= (avg_over * len(start_state_list))
            # append in reward list
            b_rewardList.append(expected_utility)
        # SARSA (fixed exploration)
        # choose random start state
        start = (random.choice([i for i in range(size)]), random.choice([i for i in range(size)]))
        source = random.choice(sourceDepotList)
        c_gridWorld.start = start
        c_gridWorld.source = source
        # learn and update optimal policy
        c_gridWorld.SARSA(gamma, alpha, epsilon)
        c_gridWorld.computePolicy(gamma, True)
        # test optimal policy (every 20 episodes)
        if (t + 1) % 20 == 0:
            expected_utility = 0
            for _ in range(avg_over):
                for taxi_x, taxi_y, pass_x, pass_y, _ in start_state_list:
                    # random start state
                    c_gridWorld.start = (taxi_x, taxi_y)
                    c_gridWorld.source = (pass_x, pass_y)
                    # get utility value
                    expected_utility += A1(c_gridWorld, 50, gamma, False)
            expected_utility /= (avg_over * len(start_state_list))
            # append in reward list
            c_rewardList.append(expected_utility)
        # SARSA (decaying exploration)
        # choose random start state
        start = (random.choice([i for i in range(size)]), random.choice([i for i in range(size)]))
        source = random.choice(sourceDepotList)
        d_gridWorld.start = start
        d_gridWorld.source = source
        # learn and update optimal policy
        d_gridWorld.SARSA(gamma, alpha, epsilon, 'decay')
        d_gridWorld.computePolicy(gamma, True)
        # test optimal policy (every 20 episodes)
        if (t + 1) % 20 == 0:
            expected_utility = 0
            for _ in range(avg_over):
                for taxi_x, taxi_y, pass_x, pass_y, _ in start_state_list:
                    # random start state
                    d_gridWorld.start = (taxi_x, taxi_y)
                    d_gridWorld.source = (pass_x, pass_y)
                    # get utility value
                    expected_utility += A1(d_gridWorld, 50, gamma, False)
            expected_utility /= (avg_over * len(start_state_list))
            # append in reward list
            d_rewardList.append(expected_utility)
    # x-axis (number of training episodes)
    xLabel = [20 * i for i in range(1, max_episodes // 20 + 1)]
    # determine maximum sum of discounted rewards reached by each algorithm
    max_discount_a = max(a_rewardList)
    max_discount_b = max(b_rewardList)
    max_discount_c = max(c_rewardList)
    max_discount_d = max(d_rewardList)
    # print max value and corresponding index
    print("Maximum average discounted reward (QL (fixed)): " + str(max_discount_a) + ", Index: " + str(a_rewardList.index(max_discount_a)))
    print("Maximum average discounted reward (QL (decay)): " + str(max_discount_b) + ", Index: " + str(b_rewardList.index(max_discount_b)))
    print("Maximum average discounted reward (SARSA (fixed)): " + str(max_discount_c) + ", Index: " + str(c_rewardList.index(max_discount_c)))
    print("Maximum average discounted reward (SARSA (decay)): " + str(max_discount_d) + ", Index: " + str(d_rewardList.index(max_discount_d)))
    # plot graphs
    if plot:
        # Q-Learning (fixed exploration)
        plt.figure(1)
        plt.plot(xLabel, a_rewardList)
        plt.xlabel("Number of Training Episodes")
        plt.ylabel("Average Discounted Reward")
        plt.title("Q-Learning (fixed exploration)")
        plt.savefig('B2_1.png')
        # Q-Learning (decaying exploration)
        plt.figure(2)
        plt.plot(xLabel, b_rewardList)
        plt.xlabel("Number of Training Episodes")
        plt.ylabel("Average Discounted Reward")
        plt.title("Q-Learning (decaying exploration)")
        plt.savefig('B2_2.png')
        # SARSA (fixed exploration)
        plt.figure(3)
        plt.plot(xLabel, c_rewardList)
        plt.xlabel("Number of Training Episodes")
        plt.ylabel("Average Discounted Reward")
        plt.title("SARSA (fixed exploration)")
        plt.savefig('B2_3.png')
        # SARSA (decaying exploration)
        plt.figure(4)
        plt.plot(xLabel, d_rewardList)
        plt.xlabel("Number of Training Episodes")
        plt.ylabel("Average Discounted Reward")
        plt.title("SARSA (decaying exploration)")
        plt.savefig('B2_4.png')
        # common plot
        plt.figure(5)
        plt.plot(xLabel, a_rewardList, label='QL (fixed)')
        plt.plot(xLabel, b_rewardList, label='QL (decay)')
        plt.plot(xLabel, c_rewardList, label='SARSA (fixed)')
        plt.plot(xLabel, d_rewardList, label='SARSA (decay)')
        plt.xlabel("Number of Training Episodes")
        plt.ylabel("Average Discounted Reward")
        plt.title("Different RL algorithms")
        plt.legend()
        plt.savefig('B2.png')
    # return best learner (based on maximum 'average' value reached)
    if max_discount_a > max(max_discount_b, max_discount_c, max_discount_d):
        return a_gridWorld
    elif max_discount_b > max(max_discount_a, max_discount_c, max_discount_d):
        return b_gridWorld
    elif max_discount_c > max(max_discount_a, max_discount_b, max_discount_d):
        return c_gridWorld
    else:
        return d_gridWorld

# max_episodes: maximum number of episodes to train RL algorithm for
# avg_over: number of episodes to use for a particular start state
def B3(max_episodes, avg_over=80):
    # define list of depots
    depotList = [(0, 0), (0, 4), (3, 0), (4, 4)]
    # define discount factor
    gamma = 0.99
    # determine best learning algorithm (B2) and compute optimal policy
    gridWorld = B2(max_episodes, False, avg_over)
    gridWorld.computePolicy(gamma, True)
    # set destination
    destination = gridWorld.destination
    # list of source locations
    sourceDepotList = [depot for depot in depotList if depot != destination]
    # complete set of initial configurations (only depot locations for start and source)
    initial_config = []
    for start in depotList:
        for source in sourceDepotList:
            initial_config.append((start, source))
    # randomly sample five start configurations for testing out the optimal policy
    test_config = random.sample(initial_config, 5)
    for idx, config in enumerate(test_config):
        start, source = config
        # set start and source locations
        gridWorld.start = start
        gridWorld.source = source
        print("Configuration: " + str(idx + 1))
        # execute optimal policy on given configuration
        A1(gridWorld, 500, gamma)
        print("__________________________________________")

# max_episodes: maximum number of episodes to train RL algorithm for
# avg_over: number of episodes to use for a particular start state
def B4(max_episodes, avg_over=80):
    # x-axis (number of episodes)
    xLabel = [50 * i for i in range(1, max_episodes // 50 + 1)]
    # define size, list of depots and wall locations
    size = 5
    depotList = [(0, 0), (0, 4), (3, 0), (4, 4)]
    walls = [((0, 0), (1, 0)), ((0, 1), (1, 1)), ((2, 0), (3, 0)), ((2, 1), (3, 1)), ((1, 3), (2, 3)), ((1, 4), (2, 4))]
    # choose a random destination location and define source depot list
    destination = random.choice(depotList)
    sourceDepotList = [depot for depot in depotList if depot != destination]
    # form list of possible start states (75 such states)
    start_state_list = []
    for depot in sourceDepotList:
        for i in range(size):
            for j in range(size):
                start_state_list.append((i, j, depot[0], depot[1], 0))
    # set discount factor
    gamma = 0.99
    # create grid world (with given destination)
    gridWorld = squareGrid(size, depotList, walls, destination)
    # set learning rate
    alpha = 0.1
    # set list of exploration rates to explore with
    epsilons = [0, 0.05, 0.1, 0.5, 0.9]
    # list of list of rewards for every epsilon
    rewardListList = []
    # list of max rewards for every epsilon
    max_reward_list = []
    # index of max reward for every epsilon
    index_list = []
    for epsilon in epsilons:
        rewardList = []
        # initialise values and policies
        gridWorld.initInstance()
        # train Q-Learning (fixed) for max_episodes
        for t in range(max_episodes):
            # random start state
            start = (random.choice([i for i in range(size)]), random.choice([i for i in range(size)]))
            source = random.choice(sourceDepotList)
            gridWorld.start = start
            gridWorld.source = source
            # update Q-values and optimal policy
            gridWorld.qLearning(gamma, alpha, epsilon)
            gridWorld.computePolicy(gamma, True)
            # test optimal policy (every 50 episodes)
            if (t + 1) % 50 == 0:
                expected_utility = 0
                for _ in range(avg_over):
                    for taxi_x, taxi_y, pass_x, pass_y, _ in start_state_list:
                        # random start state
                        gridWorld.start = (taxi_x, taxi_y)
                        gridWorld.source = (pass_x, pass_y)
                        # get utility value
                        expected_utility += A1(gridWorld, 50, gamma, False)
                expected_utility /= (avg_over * len(start_state_list))
                # append to reward list
                rewardList.append(expected_utility)
        # determine max reward and its index
        max_reward = max(rewardList)
        max_reward_list.append(max_reward)
        index_list.append(rewardList.index(max_reward))
        # append reward list to list of list of rewards
        rewardListList.append(rewardList)
    # plot graph containing all epsilons
    plt.figure(0)
    for i, epsilon in enumerate(epsilons):
        plt.plot(xLabel, rewardListList[i], label='$\epsilon =$ ' + str(epsilon))
    plt.xlabel("Number of Training Episodes")
    plt.ylabel("Average Discounted Reward")
    plt.title("Variation with Exploration Rate ($\epsilon$)")
    plt.legend()
    plt.savefig("B4_1_all.png")
    # plot graph for individual epsilon values
    for i, epsilon in enumerate(epsilons):
        plt.figure(i + 1)
        plt.plot(xLabel, rewardListList[i], label='$\epsilon =$ ' + str(epsilon))
        plt.xlabel("Number of Training Episodes")
        plt.ylabel("Average Discounted Reward")
        plt.title("Variation with Exploration Rate ($\epsilon =$" + str(epsilon) + ")")
        plt.savefig("B4_1_" + str(i + 1) + ".png")
        # print statistics
        print("Epsilon:", epsilon)
        print("Highest Average Discounted Reward: " + str(max_reward_list[i]) + ", Index: " + str(index_list[i]))
    # create another grid
    gridWorld = squareGrid(size, depotList, walls, destination)
    # set exploration rate
    epsilon = 0.1
    # set list of learning rates to experiment with
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    # list of list of rewards for every epsilon
    rewardListList = []
    # list of max rewards for every epsilon
    max_reward_list = []
    # index of max reward for every epsilon
    index_list = []
    for alpha in alphas:
        rewardList = []
        # initialise values and policies
        gridWorld.initInstance()
        # train Q-Learning (fixed) for max_episodes
        for t in range(max_episodes):
            # random start state
            start = (random.choice([i for i in range(size)]), random.choice([i for i in range(size)]))
            source = random.choice(sourceDepotList)
            gridWorld.start = start
            gridWorld.source = source
            # update Q-Value and optimal policy
            gridWorld.qLearning(gamma, alpha, epsilon)
            gridWorld.computePolicy(gamma, True)
            # test optimal policy (every 50 episodes)
            if (t + 1) % 50 == 0:
                expected_utility = 0
                for _ in range(avg_over):
                    for taxi_x, taxi_y, pass_x, pass_y, _ in start_state_list:
                        # random start state
                        gridWorld.start = (taxi_x, taxi_y)
                        gridWorld.source = (pass_x, pass_y)
                        # get utility value
                        expected_utility += A1(gridWorld, 50, gamma, False)
                expected_utility /= (avg_over * len(start_state_list))
                # append to reward list
                rewardList.append(expected_utility)
        # determine max reward and its index
        max_reward = max(rewardList)
        max_reward_list.append(max_reward)
        index_list.append(rewardList.index(max_reward))
        # append reward list to list of list of rewards
        rewardListList.append(rewardList)
    # plot graph containing all epsilons
    plt.figure(6)    
    for i, alpha in enumerate(alphas):
        plt.plot(xLabel, rewardListList[i], label='$\\alpha =$ ' + str(alpha))
    plt.xlabel("Number of Training Episodes")
    plt.ylabel("Average Discounted Reward")
    plt.title("Variation with Learning Rate ($\\alpha$)")
    plt.legend()
    plt.savefig("B4_2_all.png")
    # plot graph for individual alpha values
    for i, alpha in enumerate(alphas):
        plt.figure(i + 7)
        plt.plot(xLabel, rewardListList[i], label='$\\alpha =$ ' + str(alpha))
        plt.xlabel("Number of Training Episodes")
        plt.ylabel("Average Discounted Reward")
        plt.title("Variation with Learning Rate ($\\alpha =$" + str(alpha) + ")")
        plt.savefig("B4_2_" + str(i + 1) + ".png")
        # print statistics
        print("Alpha:", alpha)
        print("Highest Average Discounted Reward: " +  str(max_reward_list[i]) + ", Index: " + str(index_list[i]))

# max_episodes: maximum number of episodes to train RL algorithm for
# epsilon: value of exploration rate to be used
# alpha: value of learning rate to be used
# avg_over: number of episodes to use for a particular start state
def B5(max_episodes, epsilon, alpha, avg_over=80):
    # define size, list of depots and wall locations for 10-by-10 grid
    size = 10
    depotList = [(0, 1), (0, 9), (3, 6), (4, 0), (5, 9), (6, 5), (8, 9), (9, 0)]
    walls = [((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((0, 3), (1, 3)), ((2, 6), (3, 6)), ((2, 7), (3, 7)), ((2, 8), (3, 8)), ((2, 9), (3, 9)), ((3, 0), (4, 0)), ((3, 1), (4, 1)), ((3, 2), (4, 2)), ((3, 3), (4, 3)), ((5, 4), (6, 4)), ((5, 5), (6, 5)), ((5, 6), (6, 6)), ((5, 7), (6, 7)), ((7, 0), (8, 0)), ((7, 1), (8, 1)), ((7, 2), (8, 2)), ((7, 3), (8, 3)), ((7, 6), (8, 6)), ((7, 7), (8, 7)), ((7, 8), (8, 8)), ((7, 9), (8, 9))]
    # randomly sample 5 destination locations
    destinations = random.sample(depotList, k=5)
    # set discount factor
    gamma = 0.99
    # global utility is average spanning over different destinations
    global_util = 0
    # for each destination, learn and evaluate optimal policy
    for destination in destinations:
        # define possible values for source
        sourceDepotList = [depot for depot in depotList if depot != destination]
        # form list of possible start-states = (700 such states)
        start_state_list = []
        for depot in sourceDepotList:
            for i in range(size):
                for j in range(size):
                    start_state_list.append((i, j, depot[0], depot[1], 0))
        # use SARSA with decaying exploration, create grid world
        gridWorld = squareGrid(size, depotList, walls, destination)
        # initialise values and policies
        gridWorld.initInstance()
        # train model for a total of max_epsiodes
        for _ in range(max_episodes):
            # random start state
            start = (random.choice([i for i in range(size)]), random.choice([i for i in range(size)]))
            source = random.choice(sourceDepotList)
            gridWorld.start = start
            gridWorld.source = source
            # update Q-values
            gridWorld.SARSA(gamma, alpha, epsilon, 'decay', 2000)
        # compute optimal policy from Q-Values
        gridWorld.computePolicy(gamma, True)
        # test learnt policy
        expected_utility = 0
        first_time = True
        for i in range(avg_over):
            for taxi_x, taxi_y, pass_x, pass_y, _ in start_state_list:
                # random start state
                gridWorld.start = (taxi_x, taxi_y)
                gridWorld.source = (pass_x, pass_y)
                if first_time:
                    expected_utility += A1(gridWorld, 200, gamma)
                    first_time = False
                else:
                    expected_utility += A1(gridWorld, 200, gamma, False)
        expected_utility /= (avg_over * len(start_state_list))
        # update global utility
        global_util += expected_utility
        print("Average Utility:", expected_utility)
        print("________________________________________")
    # take average and print
    print("Global Average:", global_util / 5)

# max_episodes: maximum number of episodes to train RL algorithm for
# plot: boolean (whether or not to plot graphs)
# avg_over: number of episodes to use for a particular start state
def B2_low_LR(max_episodes, plot=True, avg_over=80):
    # set RL parameters (learning rate, discount factor, exploration rate)
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    # set grid size, depot and wall locations
    size = 5
    depotList = [(0, 0), (0, 4), (3, 0), (4, 4)]
    walls = [((0, 0), (1, 0)), ((0, 1), (1, 1)), ((2, 0), (3, 0)), ((2, 1), (3, 1)), ((1, 3), (2, 3)), ((1, 4), (2, 4))]
    # select destination (randomly)
    sourceDest = random.sample(depotList, 2)
    # create grid worlds for all 4 RL algorithms
    a_gridWorld = squareGrid(size, depotList, walls, sourceDest[1])
    b_gridWorld = squareGrid(size, depotList, walls, sourceDest[1])
    c_gridWorld = squareGrid(size, depotList, walls, sourceDest[1])
    d_gridWorld = squareGrid(size, depotList, walls, sourceDest[1])
    # initialise values and policies
    a_gridWorld.initInstance()
    b_gridWorld.initInstance()
    c_gridWorld.initInstance()
    d_gridWorld.initInstance()
    # possible values for source
    sourceDepotList = [depot for depot in depotList if depot != sourceDest[1]]
    # reward lists for book-keeping
    a_rewardList = []
    c_rewardList = []
    # form list of possible start states (75 such states)
    start_state_list = []
    for depot in sourceDepotList:
        for i in range(size):
            for j in range(size):
                start_state_list.append((i, j, depot[0], depot[1], 0))
    # run algorithms for max episodes
    for t in range(max_episodes):
        # Q-Learning (fixed exploration)
        # choose random start state
        start = (random.choice([i for i in range(size)]), random.choice([i for i in range(size)]))
        source = random.choice(sourceDepotList)
        a_gridWorld.start = start
        a_gridWorld.source = source
        # learn and update optimal policy
        a_gridWorld.qLearning(gamma, alpha, epsilon)
        a_gridWorld.computePolicy(gamma, True)
        # test optimal policy (every 20 episodes)
        if (t + 1) % 20 == 0:
            expected_utility = 0
            for _ in range(avg_over):
                for taxi_x, taxi_y, pass_x, pass_y, _ in start_state_list:
                    # random start state
                    a_gridWorld.start = (taxi_x, taxi_y)
                    a_gridWorld.source = (pass_x, pass_y)
                    # get utility value
                    expected_utility += A1(a_gridWorld, 50, gamma, False)
            expected_utility /= (avg_over * len(start_state_list))
            # append in reward list
            a_rewardList.append(expected_utility)
        # SARSA (fixed exploration)
        # choose random start state
        start = (random.choice([i for i in range(size)]), random.choice([i for i in range(size)]))
        source = random.choice(sourceDepotList)
        c_gridWorld.start = start
        c_gridWorld.source = source
        # learn and update optimal policy
        c_gridWorld.SARSA(gamma, alpha, epsilon)
        c_gridWorld.computePolicy(gamma, True)
        # test optimal policy (every 20 episodes)
        if (t + 1) % 20 == 0:
            expected_utility = 0
            for _ in range(avg_over):
                for taxi_x, taxi_y, pass_x, pass_y, _ in start_state_list:
                    # random start state
                    c_gridWorld.start = (taxi_x, taxi_y)
                    c_gridWorld.source = (pass_x, pass_y)
                    # get utility value
                    expected_utility += A1(c_gridWorld, 50, gamma, False)
            expected_utility /= (avg_over * len(start_state_list))
            # append in reward list
            c_rewardList.append(expected_utility)
    # x-axis (number of training episodes)
    xLabel = [20 * i for i in range(1, max_episodes // 20 + 1)]
    # determine maximum sum of discounted rewards reached by each algorithm
    max_discount_a = max(a_rewardList)
    max_discount_c = max(c_rewardList)
    # print max value and corresponding index
    print("Maximum average discounted reward (QL (fixed)): " + str(max_discount_a) + ", Index: " + str(a_rewardList.index(max_discount_a)))
    print("Maximum average discounted reward (SARSA (fixed)): " + str(max_discount_c) + ", Index: " + str(c_rewardList.index(max_discount_c)))
    # plot graphs
    if plot:
        # Q-Learning (fixed exploration)
        plt.figure(1)
        plt.plot(xLabel, a_rewardList)
        plt.xlabel("Number of Training Episodes")
        plt.ylabel("Average Discounted Reward")
        plt.title("Q-Learning (fixed exploration)")
        plt.savefig('B2_low_LR_Q.png')
        # SARSA (fixed exploration)
        plt.figure(3)
        plt.plot(xLabel, c_rewardList)
        plt.xlabel("Number of Training Episodes")
        plt.ylabel("Average Discounted Reward")
        plt.title("SARSA (fixed exploration)")
        plt.savefig('B2_low_LR_S.png')

# driver function: main()
def main():
    # MDP or RL
    isMarkov = True if sys.argv[1] == 'MDP' else False
    if isMarkov:
        # MDP: A, get sub-part
        sub_part = int(sys.argv[2])
        if sub_part == 1:
            # example run (using value-iteration procedure)
            A2('a', (3, 0), (0, 0), (4, 4), False)
        elif sub_part == 2:
            # Value-Iteration, get sub-sub-part
            sub_sub_part = sys.argv[3]
            if sub_sub_part == 'a':
                if sys.argv[4] == 'default':
                    A2('a')
                else:
                    start = (int(sys.argv[5]), int(sys.argv[6]))
                    source = (int(sys.argv[7]), int(sys.argv[8]))
                    destination = (int(sys.argv[9]), int(sys.argv[10]))
                    epsilon = float(sys.argv[11])
                    A2('a', start, source, destination, True, epsilon)
            elif sub_sub_part == 'b':
                if sys.argv[4] == 'default':
                    A2('b')
                else:
                    start = (int(sys.argv[5]), int(sys.argv[6]))
                    source = (int(sys.argv[7]), int(sys.argv[8]))
                    destination = (int(sys.argv[9]), int(sys.argv[10]))
                    A2('a', start, source, destination)
            else:
                # part (c)
                A2('c')
            exit()
        else:
            # part 3
            A3()
    else:
        # RL: B, get sub-part
        sub_part = int(sys.argv[2])
        if sub_part == 2:
            B2(2000)
        elif sub_part == 3:
            B3(2000)
        elif sub_part == 4:
            B4(4000)
        else:
            # part 5
            B5(10000, 0.5, 0.2)

main()
#B2_low_LR(2000)