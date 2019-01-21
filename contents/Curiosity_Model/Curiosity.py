"""This is a simple implementation of [Large-Scale Study of Curiosity-Driven Learning](https://arxiv.org/abs/1808.04355)"""

import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt


class CuriosityNet:
    def __init__(
            self,
            n_a,
            n_s,
            lr=0.01,
            gamma=0.98,
            epsilon=0.95,
            replace_target_iter=300,
            memory_size=10000,
            batch_size=128,
            output_graph=False,
    ):
        self.n_a = n_a
        self.n_s = n_s
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size

        # total learning step
        self.learn_step_counter = 0
        self.memory_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_s * 2 + 2))
        self.tfs, self.tfa, self.tfr, self.tfs_, self.dyn_train, self.dqn_train, self.q, self.int_r = \
            self._build_nets()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_nets(self):
        tfs = tf.placeholder(tf.float32, [None, self.n_s], name="s")    # input State
        tfa = tf.placeholder(tf.int32, [None, ], name="a")              # input Action
        tfr = tf.placeholder(tf.float32, [None, ], name="ext_r")        # extrinsic reward
        tfs_ = tf.placeholder(tf.float32, [None, self.n_s], name="s_")  # input Next State

        # dynamics net
        dyn_s_, curiosity, dyn_train = self._build_dynamics_net(tfs, tfa, tfs_)

        # normal RL model
        total_reward = tf.add(curiosity, tfr, name="total_r")
        q, dqn_loss, dqn_train = self._build_dqn(tfs, tfa, total_reward, tfs_)
        return tfs, tfa, tfr, tfs_, dyn_train, dqn_train, q, curiosity

    def _build_dynamics_net(self, s, a, s_):
        with tf.variable_scope("dyn_net"):
            float_a = tf.expand_dims(tf.cast(a, dtype=tf.float32, name="float_a"), axis=1, name="2d_a")
            sa = tf.concat((s, float_a), axis=1, name="sa")
            encoded_s_ = s_                # here we use s_ as the encoded s_

            dyn_l = tf.layers.dense(sa, 32, activation=tf.nn.relu)
            dyn_s_ = tf.layers.dense(dyn_l, self.n_s)  # predicted s_
        with tf.name_scope("int_r"):
            squared_diff = tf.reduce_sum(tf.square(encoded_s_ - dyn_s_), axis=1)  # intrinsic reward

        # It is better to reduce the learning rate in order to stay curious
        train_op = tf.train.RMSPropOptimizer(self.lr, name="dyn_opt").minimize(tf.reduce_mean(squared_diff))
        return dyn_s_, squared_diff, train_op

    def _build_dqn(self, s, a, r, s_):
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(s, 128, tf.nn.relu)
            q = tf.layers.dense(e1, self.n_a, name="q")
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(s_, 128, tf.nn.relu)
            q_ = tf.layers.dense(t1, self.n_a, name="q_")

        with tf.variable_scope('q_target'):
            q_target = r + self.gamma * tf.reduce_max(q_, axis=1, name="Qmax_s_")

        with tf.variable_scope('q_wrt_a'):
            a_indices = tf.stack([tf.range(tf.shape(a)[0], dtype=tf.int32), a], axis=1)
            q_wrt_a = tf.gather_nd(params=q, indices=a_indices)

        loss = tf.losses.mean_squared_error(labels=q_target, predictions=q_wrt_a)   # TD error
        train_op = tf.train.RMSPropOptimizer(self.lr, name="dqn_opt").minimize(
            loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "eval_net"))
        return q, loss, train_op

    def store_transition(self, s, a, r, s_):            
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        s = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q, feed_dict={self.tfs: s})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_a)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)

        # sample batch memory from all memory
        top = self.memory_size if self.memory_counter > self.memory_size else self.memory_counter
        sample_index = np.random.choice(top, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        bs, ba, br, bs_ = batch_memory[:, :self.n_s], batch_memory[:, self.n_s], \
            batch_memory[:, self.n_s + 1], batch_memory[:, -self.n_s:]
        self.sess.run(self.dqn_train, feed_dict={self.tfs: bs, self.tfa: ba, self.tfr: br, self.tfs_: bs_})
        if self.learn_step_counter % 1000 == 0:     # delay training in order to stay curious
            self.sess.run(self.dyn_train, feed_dict={self.tfs: bs, self.tfa: ba, self.tfs_: bs_})
        self.learn_step_counter += 1


env = gym.make('MountainCar-v0')
env = env.unwrapped

dqn = CuriosityNet(n_a=3, n_s=2, lr=0.01, output_graph=False)
ep_steps = []
for epi in range(200):
    s = env.reset()
    steps = 0
    while True:
        env.render()
        a = dqn.choose_action(s)
        s_, r, done, info = env.step(a)
        dqn.store_transition(s, a, r, s_)
        dqn.learn()
        if done:
            print('Epi: ', epi, "| steps: ", steps)
            ep_steps.append(steps)
            break
        s = s_
        steps += 1

plt.plot(ep_steps)
plt.ylabel("steps")
plt.xlabel("episode")
plt.show()