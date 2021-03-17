# Maze
A simple SARSA &amp; Q-Learning Reinforcement Learning agent

This reinforcement learning agent is based on using either the SARSA or Q-Learning algorithms:
- SARSA: [https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action](url)
- Q Learning: [https://en.wikipedia.org/wiki/Q-learning](url)

The aim of the agent is to get from its start position in the top-left corner of the maze, to the terminal state in the bottom right corner of the maze. The agent can only 'walk' on white grid squares, with all grey squares denoting the 'cliff'. Falling off the cliff will send the agent back to the start of the maze.
