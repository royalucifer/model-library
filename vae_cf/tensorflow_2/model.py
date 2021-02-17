import tensorflow as tf
import tensorflow.keras as tfk


class VAE(tf.keras.Model):
    def __init__(self, p_dims, q_dims=None, drop_prob=0, l2_reg=0, seed=None):
        super(VAE, self).__init__()
        self.seed = seed
        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else:
            assert q_dims[0] == p_dims[-1], 'Input and output dimension must equal.'
            assert q_dims[-1] == p_dims[0], 'Latent dimension for p and q mismatches.'
            self.q_dims = q_dims
        self.input_dim = self.q_dims[0]
        self.latent_dim = self.q_dims[-1]

        l2_regularizer = tfk.regularizers.l2(l2_reg)
        w_initializer = tfk.initializers.GlorotUniform(self.seed)
        b_initializer = tfk.initializers.TruncatedNormal(stddev=0.001, seed=self.seed)

        self.inference_net = tfk.Sequential([
            tfk.layers.InputLayer(input_shape=(self.input_dim,)),
            tfk.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)),
            tfk.layers.Dropout(drop_prob)
        ])
        for dim in self.q_dims[1:]:
            if dim != self.q_dims[-1]:
                dense_layer = tfk.layers.Dense(
                    dim, 'tanh',
                    kernel_initializer=w_initializer,
                    bias_initializer=b_initializer,
                    kernel_regularizer=l2_regularizer,
                    bias_regularizer=l2_regularizer)
            else:
                dense_layer = tfk.layers.Dense(
                    dim*2,
                    kernel_initializer=w_initializer,
                    bias_initializer=b_initializer,
                    kernel_regularizer=l2_regularizer,
                    bias_regularizer=l2_regularizer)
            self.inference_net.add(dense_layer)

        self.generative_net = tfk.Sequential([
            tfk.layers.InputLayer(input_shape=(self.latent_dim,)),
        ])
        for dim in self.p_dims[1:]:
            if dim != self.p_dims[-1]:
                dense_layer = tfk.layers.Dense(
                    dim, 'tanh',
                    kernel_initializer=w_initializer,
                    bias_initializer=b_initializer,
                    kernel_regularizer=l2_regularizer,
                    bias_regularizer=l2_regularizer)
            else:
                dense_layer = tfk.layers.Dense(
                    dim,
                    kernel_initializer=w_initializer,
                    bias_initializer=b_initializer,
                    kernel_regularizer=l2_regularizer,
                    bias_regularizer=l2_regularizer)
            self.generative_net.add(dense_layer)

    def encode(self, x, training=True):
        h = self.inference_net(x, training=training)
        mean = h[:, :self.q_dims[-1]]
        logvar = h[:, self.q_dims[-1]:]
        KL = tf.reduce_mean(tf.reduce_sum(
            0.5 * (-logvar - 1 + tf.exp(logvar) + mean**2), axis=1))
        return mean, logvar, KL

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    @tf.function
    def call(self, inputs):
        mean, _, _ = self.encode(inputs, training=False)
        return {'logits': self.decode(mean)}
