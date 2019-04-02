"""
Sarsa is a online updating method for Reinforcement learning.

Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.

You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""

from maze_env import Maze
from RL_brain import SarsaTable


def update():
    for episode in range(100):
        # initial state
        state = env.reset()

        # RL choose action based on state
        action = RL.choose_action(str(state))

        while True:
            # fresh env
            env.render()

            # RL take action and get next state and reward
            state_, reward, done = env.step(action)

            # RL choose action based on next state
            # 和Q-learning不同的地方就在这里，Q-learning选择Q表中最大的那一项，可是实际在S_时不一定这么走
            # 而这个直接是根据S_选择了a_并且一定这么走
            action_ = RL.choose_action(str(state_))

            # RL learn from this transition (s, a, r, s_, a_) ==> Sarsa
            RL.learn(str(state), action, reward, str(state_), action_)

            # swap state and action
            state = state_
            action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()