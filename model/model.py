import tensorflow as tf
import numpy as np
import pandas as pd

class HParams:
    def __init__(self, n_vocab=0, n_ctx=1024, n_embd=768, n_head=12, n_layer=12):
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer

def default_hparams():
    return HParams()

def shape_list(x):
    """Deal with dynamic shape in TensorFlow or NumPy or pandas."""
    if isinstance(x, (tf.Tensor, np.ndarray)):
        static = x.shape.as_list() if isinstance(x, tf.Tensor) else list(x.shape)
        dynamic = tf.shape(x) if isinstance(x, tf.Tensor) else list(x.shape)
        return [dynamic[i] if s is None else s for i, s in enumerate(static)]
    elif isinstance(x, pd.Series):
        return [len(x)]
    else:
        raise ValueError("Unsupported type for shape_list: {}".format(type(x)))



def gelu(x):
    x = tf.cast(x, dtype=tf.float32)  # Cast x to float32
    return 0.5 * x * (1 + tf.math.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean=0, std=1, then do a diagonal affine transform."""
    with tf.name_scope(scope):
        n_state = x.shape[-1]
        g = tf.Variable(tf.constant(1.0, shape=[n_state]), name='g', dtype=tf.float32)
        b = tf.Variable(tf.constant(0.0, shape=[n_state]), name='b', dtype=tf.float32)
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
        x = (x - u) * tf.math.rsqrt(s + epsilon)
        x = tf.cast(x, dtype=tf.float32)  # Explicitly cast to float
        x = x * g + b
        return x


def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m // n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a * b])

def conv1d(x, name, nf, rf=1, w_init=tf.random_normal_initializer(0.02), b_init=tf.constant_initializer(0), trainable=True):
    nx = x.shape[-1]  # Size of the last dimension in x
    assert nx is not None, f'Inputs to {name} should have defined dimensions, but received {x}'

    w = tf.Variable(initial_value=w_init(shape=(1, rf, nx, nf), dtype=tf.float32), name=f'{name}_w', trainable=trainable)
    b = tf.Variable(initial_value=b_init(shape=(nf,), dtype=tf.float32), name=f'{name}_b', trainable=trainable)

    # Use tf.matmul for matrix multiplication
    c = tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf])) + b

    # Use tf.reshape to reshape the result
    c = tf.reshape(c, tf.concat([tf.shape(x)[:-1], [nf]], axis=0))
    
    return c





def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner."""
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)

def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - tf.constant(1e10, dtype=w.dtype) * (1 - b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.math.rsqrt(tf.cast(v.shape[-1], w.dtype))

        w = mask_attn_weights(w)
        w = tf.nn.softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.name_scope(scope):
        c = conv1d(x, 'c_attn', n_state * 3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present

def mlp(x, scope, n_state, *, hparams):
    with tf.name_scope(scope):
        nx = x.shape[-1]
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2

def block(x, scope, *, past, hparams):
    with tf.name_scope(scope):
        nx = x.shape[-1]
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        
        # Ensure all tensors are of type tf.float32
        x = tf.cast(x, dtype=tf.float32)
        a = tf.cast(a, dtype=tf.float32)

        # Use tf.math.add for element-wise addition
        x = tf.math.add(x, a)  
        
        m = mlp(norm(x, 'ln_2'), 'mlp', nx * 4, hparams=hparams)
        
        # Ensure m is of type tf.float32
        m = tf.cast(m, dtype=tf.float32)
        
        # Use tf.math.add for element-wise addition
        x = tf.math.add(x, m)  

        return x, present


def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1] * ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0] if len(tokens.shape) > 1 else 1
    nsteps = tf.shape(tokens)[1] if len(tokens.shape) > 1 else 1
    return expand_tile(past_length + tf.range(nsteps), batch_size)

def binary_classifier_model(hparams, X, past=None, scope='model', reuse=False):
    with tf.name_scope(scope):
        results = {}
        shape = shape_list(X)
        batch = shape[0]

        # Determine the vocabulary size dynamically
        vocab_size = tf.reduce_max(X) + 1

        wpe = tf.Variable(tf.random.normal([hparams.n_ctx, hparams.n_embd], stddev=0.01), name='wpe')
        wte = tf.Variable(tf.random.normal([vocab_size, hparams.n_embd], stddev=0.02), name='wte')
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')

        # Binary classification
        h_flat = tf.reduce_mean(h, axis=1)  # Global average pooling
        logits = conv1d(h_flat, 'classifier', 1)
        logits = tf.squeeze(logits, axis=1)
        results['logits'] = logits
        return results