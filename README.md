# Maze
A simple SARSA &amp; Q-Learning Reinforcement Learning agent inspired by Example 6.6 in "Reinforcement Learning: An Introduction" (Sutton & Barto 2017)

This reinforcement learning agent is based on using either the SARSA or Q-Learning algorithms:
- SARSA: [https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action](url)
- Q Learning: [https://en.wikipedia.org/wiki/Q-learning](url)

The aim of the agent is to get from its start position in the top-left corner of the maze, to the terminal state in the bottom right corner of the maze. The agent can only 'walk' on white grid squares, with all grey squares denoting the 'cliff'. Falling off the cliff will send the agent back to the start of the maze.

Two version of this maze have been created:
- The 1st version is the simpler. Here the agent is simply looking to reach the end of the maze without falling off the cliff
- The 2nd iteration of the maze is extended to allow for the agent to try and collect a coin for which it earns a reward. The coin is denoted by a green square

Extensions still to be considered will be to expand the maze to include multiple coins, traps to avoid, and to implement an adaptive epsilon-greedy search method that reduces the levels of exploration over time.
