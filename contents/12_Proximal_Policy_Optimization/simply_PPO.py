"""
A simple version of Proximal Policy Optimization (PPO) using single thread.

Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [http://adsabs.harvard.edu/abs/2017arXiv170702286H]
2. Proximal Policy Optimization Algorithms (OpenAI): [http://adsabs.harvard.edu/abs/2017arXiv170706347S]

View more on my tutorial website: https://morvanzhou.github.io/tutorials

Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
from tensorflow.contrib.distributions import Normal, kl_divergence
import numpy as np
import matplotlib.pyplot as plt
import gym

EP_MAX = 800
EP_STEP = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
KL_TARGET = 0.01
T = 100


class PPO(object):
    lam = 0.5
    sess = tf.Session()

    def __init__(self, s_dim, a_dim, kl_target,):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.kl_target = kl_target

        self.tfs = tf.placeholder(tf.float32, [None, s_dim])

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, ])
            self.advantage = self.tfdc_r - tf.squeeze(self.v)
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)

        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.sample_op = pi.sample(1)
        self.tfa = tf.placeholder(tf.float32, [None, ], 'action')
        with tf.variable_scope('ratio'):
            # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
            ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
        with tf.variable_scope('kl'):
            self.kl = tf.stop_gradient(tf.reduce_mean(kl_divergence(oldpi, pi)))
        self.tflam = tf.placeholder(tf.float32, None, 'lambda')
        self.tfadv = tf.placeholder(tf.float32, [None, ], 'advantage')
        with tf.variable_scope('loss'):
            self.aloss = -(tf.reduce_mean(ratio * self.tfadv) - self.tflam * self.kl)
        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update_oldpi(self):
        self.sess.run(self.update_oldpi_op)

    def update(self, s, a, r, m=20, b=10):
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        for _ in range(m):
            _, kl = self.sess.run(
                [self.atrain_op, self.kl],
                {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: self.lam})
            if kl > 4*KL_TARGET:
                break

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(b)]

        # adaptive lambda
        if kl < self.kl_target / 1.5:
            self.lam /= 2
        elif kl > self.kl_target * 1.5:
            self.lam *= 2
        self.lam = np.clip(self.lam, 1e-4, 10)

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, self.a_dim, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, self.a_dim, tf.nn.softplus, trainable=trainable)
            norm_dist = Normal(loc=tf.squeeze(mu), scale=tf.squeeze(sigma))
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

env = gym.make('Pendulum-v0').unwrapped
ppo = PPO(3, 1, KL_TARGET)
all_ep_r = []

for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(1, EP_STEP):    # one episode
        env.render()
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        s = s_
        ep_r += r

        # update ppo
        if t % (T-1) == 0 or t == EP_STEP-1:
            ppo.update_oldpi()
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.concatenate(buffer_a), np.array(discounted_r)
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br, m=A_UPDATE_STEPS, b=C_UPDATE_STEPS)
    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        "|lamb: %.3f" % ppo.lam,
    )

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()