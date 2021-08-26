import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib.animation import FuncAnimation

class CliffWorld():


    def __init__(self, gridHeight = 6, gridWidth = 30, epsilon = 0.1, 
                alpha=0.5, gamma = 1, startPosition = (0,0), endPosition = (5,29), 
                theCoins = [(2,23)],
                theCliff = [(5,0),(5,1),(5,8),(5,9),(5,10),(5,11),(5,12),(5,13),(5,14),(5,15),(5,16),(5,17),
                            (4,10),(4,11),(4,12),(4,13),(4,14),
                            (3,5),(3,12),(3,13),(3,14),(3,20),(3,26),
                            (2,5),(2,12),(2,13),(2,14),(2,20),(2,26),(2,27),
                            (1,4),(1,5),(1,18),(1,19),(1,26),(1,27),(1,28),
                            (0,2),(0,3),(0,4),(0,23),(0,24),(0,25),(0,26),(0,27),(0,28),(0,29)]): 

        self.gridHeight = gridHeight
        self.gridWidth = gridWidth
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.startPosition = startPosition
        self.endPosition = endPosition
        self.theCoins = theCoins
        self.theCliff = theCliff
        self.possibleActions = ['up', 'down', 'left', 'right']
        self.initialiseQEstimate()


    def initialiseQEstimate(self):

        self.SarsaQEstimate = {}
        self.QLearningQEstimate = {}

        for row in range(self.gridHeight):
            for column in range(self.gridWidth):
                self.SarsaQEstimate[(row,column)] = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
                self.QLearningQEstimate[(row,column)] = {'up': 0, 'down': 0, 'left': 0, 'right': 0}


    def epsilonGreedyAction(self, state, qEstimate, epsilon):
        """
        A function to select an action from a given state using the epsilon-greedy  selection method
        """

        import numpy as np

        if np.random.rand() < epsilon:
            # This will give an action between 0 and 3 which relates to up, down, left, right
            action = np.random.choice(self.possibleActions)
        else:
            # If we do not pick a random action then we must pick the action from that grid position with the highest estimated Q value
            # Create a list of the actions that have the highest Q Estimate, and then solve any tie-breaks randomly
            stateQEstimates = qEstimate[state]
            bestActions = [action for action, value in stateQEstimates.items() if value == max(stateQEstimates.values())]
            action = np.random.choice(bestActions)

        return action


    def takeAction(self, currentState, action):
        
        row = currentState[0]
        column = currentState[1]

        if action == 'up':
            newState = (row - 1, column)
        elif action == 'down':
            newState = (row + 1, column)
        elif action == 'left':
            newState = (row, column - 1)
        elif action == 'right':
            newState = (row, column + 1)
        else:
            raise NameError(action, 'This is not a valid Action!')

        if newState in self.theCliff:
            reward = -100
            newState = self.startPosition
        elif (newState[0] not in range(self.gridHeight)) or (newState[1] not in range(self.gridWidth)):
            reward = -1
            newState = currentState
        elif newState in self.theCoins:
            reward = -1
        elif newState == self.endPosition:
            reward = 0
        else:
            reward = -1

        return reward, newState


    def sarsaUpdateRule(self, currentState, currentAction, earnedReward, nextState, nextAction):

        nextQEstimate = self.SarsaQEstimate[nextState][nextAction]

        self.SarsaQEstimate[currentState][currentAction] += self.alpha * (earnedReward + self.gamma * nextQEstimate - self.SarsaQEstimate[currentState][currentAction] )


    def qLearningUpdateRule(self, currentState, currentAction, earnedReward, nextState):
        
        import numpy as np
        
        nextStateQEstimates = self.QLearningQEstimate[nextState]
        possibleNextActions = [action for action, value in nextStateQEstimates.items() if value == max(nextStateQEstimates.values())]
        nextAction = np.random.choice(possibleNextActions) 
        nextQEstimate = self.QLearningQEstimate[nextState][nextAction]
        
        self.QLearningQEstimate[currentState][currentAction] += self.alpha * (earnedReward + self.gamma * nextQEstimate - self.QLearningQEstimate[currentState][currentAction] )        


    def runSarsaLearning(self, epsilon):
        """
        A function to calculate the total reward earned by the agent in the Cliff world following the SARSA process in a single episode
        Q(S(t), A(t)) <-- Q(S(t), A(t)) + alpha * [R(t+1) + gamma * Q(S(t+1), A(t+1)) -Q(S(t), A(t))]
        """

        currentState = self.startPosition        
        yValues = [currentState[0]+0.5]
        xValues = [currentState[1]+0.5]

        while currentState != self.endPosition:
            currentAction = self.epsilonGreedyAction(currentState, self.SarsaQEstimate, epsilon)
            earnedReward, nextState = self.takeAction(currentState, currentAction)
            xValues.append(nextState[1]+0.5)
            yValues.append(nextState[0]+0.5)
            nextAction = self.epsilonGreedyAction(nextState, self.SarsaQEstimate, epsilon)
            self.sarsaUpdateRule(currentState, currentAction, earnedReward, nextState, nextAction)
            currentState = nextState
        
        return xValues, yValues


    def runQLearning(self, epsilon):
        """
        A function to calculate the total reward earned by the agent in the Cliff world following the Q-Learning process in a single episode
        Q(S(t), A(t)) <-- Q(S(t), A(t)) + alpha * [R(t+1) + gamma * Max[Q(S(t+1), A)] -Q(S(t), A(t))]
        """
        
        currentState = self.startPosition
        yValues = [currentState[0]+0.5]
        xValues = [currentState[1]+0.5]

        while currentState != self.endPosition:
            currentAction = self.epsilonGreedyAction(currentState, self.QLearningQEstimate, epsilon)
            earnedReward, nextState = self.takeAction(currentState, currentAction)
            xValues.append(nextState[1]+0.5)
            yValues.append(nextState[0]+0.5)
            self.qLearningUpdateRule(currentState, currentAction, earnedReward, nextState)
            
            currentState = nextState
        
        return xValues, yValues

########################################################################################################################################################

def plotDumbCliffLearning(epsilon = 0.1):
    
    agent = CliffWorld(epsilon = epsilon)
    xValues, yValues = agent.runSarsaLearning(epsilon = epsilon)

    fig = plt.figure()

    plt.xlim(0, 30, 1)
    plt.ylim(0, 6, 1)
    plt.title('Dumb Cliff world')

    graph, = plt.plot([], [], 'o', markersize=12, color = 'r')
    plt.grid(which='major', axis='both', linestyle='-', color='grey', linewidth=1, markevery=1)
    img = plt.imread('/Users/Tom/Documents/DataScience/LiverpoolUni/COMP532/Assignment1/gameWithCoins.png')
    plt.imshow(img, extent=[0,30,0,6])


    def animate(i):
        graph.set_data(xValues[i], 6 - yValues[i])
        return graph

    animation = FuncAnimation(fig, animate, frames = len(xValues), interval = 20)
    
    plt.show()


def plotSmartCliffLearning(epsilon = 0.1, episodes = 5):
    
    agent = CliffWorld(epsilon = epsilon)

    for epoch in range(episodes):
        print('Running epoch {}'.format(epoch))
        agent.runSarsaLearning(epsilon = epsilon)

    xValues, yValues = agent.runSarsaLearning(epsilon = 0)

    fig = plt.figure()

    plt.xlim(0, 30, 1)
    plt.ylim(0, 6, 1)
    plt.title('Smart Cliff world')

    graph, = plt.plot([], [], 'o', markersize=12, color = 'r')
    plt.grid(which='major', axis='both', linestyle='-', color='grey', linewidth=1, markevery=1)
    img = plt.imread('/Users/Tom/Documents/DataScience/LiverpoolUni/COMP532/Assignment1/gameWithCoins.png')
    plt.imshow(img, extent=[0,30,0,6])


    def animate(i):
        graph.set_data(xValues[i], 6 - yValues[i])
        return graph

    animation = FuncAnimation(fig, animate, frames = len(xValues), interval = 20)
    
    plt.show()


def plotWorld():
    data = np.array([[-1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1],
                    [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,0],
                    [0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0,0,3,0,0,1,1,0,0],
                    [0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,2]])

    cmap = colors.ListedColormap(['grey', 'white', 'black', 'gold', 'green'])
    bounds = [0,0.9,1.9,2.9]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    ax.imshow(data, cmap = cmap)
    ax.grid(which='major', axis='both', linestyle='-', color='grey', linewidth=1)
    ax.set_yticks(np.arange(-0.5,5,1))
    ax.set_xticks(np.arange(-0.5,30,1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.title('Cliff world')
    plt.show()

if __name__ == '__main__':
    #plotWorld()
    #plotDumbCliffLearning()
    plotSmartCliffLearning(episodes = 5000)