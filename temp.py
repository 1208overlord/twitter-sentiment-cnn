import tensorflow as tf
with tf.device('/{}:{}'.format('gpu','0')):
    x = [0., -1., 2., 3.]
    softmax_x = tf.nn.softmax(x)
    session = tf.Session()
    print(session.run(softmax_x))