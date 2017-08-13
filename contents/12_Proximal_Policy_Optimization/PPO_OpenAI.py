"""
A simple version of OpenAI's Proximal Policy Optimization (PPO). [http://adsabs.harvard.edu/abs/2017arXiv170706347S]
Distributing workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated. I think A3C may be faster than this version of PPO, because this PPO has to stop
parallel data-collection for training.

View more on my tutorial website: https://morvanzhou.github.io/tutorials

Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
from tensorflow.contrib.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import gym, threading
from queue import Queue

EP_MAX = 600
EP_LEN = 200
N_WORKER = 3
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
ROLL_OUT_STEP = 32
UPDATE_STEP = 10
EPSILON = 0.2                # Clipped surrogate objective
S_DIM, A_DIM = 3, 1


class PPO(object):
    def __init__(self, s_dim, a_dim,):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.sess = tf.Session()

        self.tfs = tf.placeholder(tf.float32, [None, s_dim], 'state')

        # critic
        l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
        self.v = tf.layers.dense(l1, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # choosing action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, a_dim], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv   # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())

    def update(self, coord, queue, rolling_events):
        while not coord.should_stop():
            if queue.full():
                self.sess.run(self.update_oldpi_op)   # old pi to pi

                data = [queue.get() for _ in range(queue.qsize())]
                data = np.vstack(data)
                s, a, r = data[:, :self.s_dim], data[:, self.s_dim: self.s_dim + self.a_dim], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]

                [re.set() for re in rolling_events]     # set roll-out available

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 200, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, self.a_dim, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, self.a_dim, tf.nn.softplus, trainable=trainable)
            norm_dist = Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


class Worker(object):
    def __init__(self, globalPPO, roll_out_steps, wid, game, ep_len, rolling_event):
        self.roll_out_steps = roll_out_steps
        self.wid = wid
        self.ep_len = ep_len
        self.rolling_event = rolling_event
        self.env = gym.make(game).unwrapped
        self.ppo = globalPPO

    def work(self, coord, queue,):
        global GLOBAL_EP, GLOBAL_RUNNING_R
        while not coord.should_stop():
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(self.ep_len):
                a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)  # normalize reward, find to be useful
                s = s_
                ep_r += r

                # get update buffer
                if (t+1) % self.roll_out_steps == 0 or t == self.ep_len - 1:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []           # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    queue.put(np.hstack((bs, ba, br)))
                    if GLOBAL_EP >= EP_MAX:             # stop training
                        coord.request_stop()
                        break
                    else:
                        self.rolling_event.clear()      # stop roll-out
                        self.rolling_event.wait()       # stop and wait until network is updated

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0: GLOBAL_RUNNING_R.append(ep_r)
            else: GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1]*0.9+ep_r*0.1)
            GLOBAL_EP += 1
            print('W%i' % self.wid, '|Ep: %i' % GLOBAL_EP, '|Ep_r: %.2f' % ep_r,)


if __name__ == '__main__':
    globalPPO = PPO(S_DIM, A_DIM)
    workers = [Worker(
        globalPPO=globalPPO, roll_out_steps=ROLL_OUT_STEP, wid=i, game='Pendulum-v0',
        ep_len=EP_LEN, rolling_event=threading.Event()) for i in range(N_WORKER)]

    GLOBAL_EP = 0
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()
    QUEUE = Queue(maxsize=N_WORKER)
    threads = []
    for worker in workers:  # worker threads
        t = threading.Thread(target=worker.work, args=(COORD, QUEUE))
        t.start()
        threads.append(t)
    # update thread for network
    threads.append(threading.Thread(target=globalPPO.update, args=(COORD, QUEUE, [w.rolling_event for w in workers])))
    threads[-1].start()
    COORD.join(threads)

    # plot reward change
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode'); plt.ylabel('Moving reward'); plt.ion(); plt.show()

    env = gym.make('Pendulum-v0')       # testing
    while True:
        s = env.reset()
        for t in range(400):
            env.render()
            a = globalPPO.choose_action(s)
            s = env.step(a)[0]