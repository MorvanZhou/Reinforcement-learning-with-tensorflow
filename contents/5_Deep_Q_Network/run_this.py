from maze_env import Maze
from RL_brain import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(300):
        # initial state
        state = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on state
            action = RL.choose_action(state)

            # RL take action and get next state and reward
            state_, reward, done = env.step(action)

            RL.store_transition(state, action, reward, state_)

            # 结合learn方法的实现来看，tf.session.run方法中GPU加速的部分应该是矩阵运算.
            # 而且应该不是异步方法，由于run方法只跑一趟只进行一次参数更新，所以这里的learn方法实际也就只进行了一次参数更新
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap state
            state = state_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()