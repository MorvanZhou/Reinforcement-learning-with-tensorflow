import tensorflow as tf
from tensorflow.contrib.distributions import Normal, kl_divergence
import numpy as np
import gym

EP_MAX = 1000
EP_STEP = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.001
M = 20
B = 10
KL_TARGET = 0.01
T = 200


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
            self.t = ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
            # ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)

        with tf.variable_scope('kl'):
            self.kl = tf.stop_gradient(tf.reduce_mean(kl_divergence(oldpi, pi)))
        self.tflam = tf.placeholder(tf.float32, None, 'lambda')
        self.tfadv = tf.placeholder(tf.float32, [None, ], 'advantage')
        with tf.variable_scope('entropy'):
            entropy = tf.stop_gradient(pi.entropy())
        with tf.variable_scope('loss'):
            self.aloss = -(tf.reduce_mean(ratio * self.tfadv) - self.tflam * self.kl + 0.2 * entropy)
        # self.aloss = -(tf.reduce_mean(pi.log_prob(self.tfa)*self.tfadv) + 0.2 * tf.stop_gradient(pi.entropy()))
        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update_oldpi(self):
        self.sess.run(self.update_oldpi_op)

    def update(self, s, a, r, m=20, b=10):
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})

        # update actor
        for _ in range(m):
            _, kl, t = self.sess.run(
                [self.atrain_op, self.kl, self.t],
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
        return kl

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
        return self.sess.run(self.v, {self.tfs: s})

env = gym.make('Pendulum-v0').unwrapped
ppo = PPO(3, 1, KL_TARGET)

for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(1, EP_STEP): # one episode
        env.render()
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)
        if t == EP_STEP-1: done = True
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append(r)
        s = s_
        ep_r += r
        if t % (T-1) == 0 or done:
            ppo.update_oldpi()
            if done:
                v_s_ = 0
            else:
                v_s_ = ppo.get_v(s_)[0, 0]
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.concatenate(buffer_a), np.array(discounted_r)
            buffer_s, buffer_a, buffer_r = [], [], []
            kl = ppo.update(bs, ba, br, m=M, b=B)

    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        "|KL: %.3f" % kl,
        "|lamb: %.3f" % ppo.lam,
    )