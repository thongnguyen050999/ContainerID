import tensorflow as tf

tf.compat.v1.disable_eager_execution()


checkpoint = '/home/tony/ViMusic/models/remi/REMI-tempo-chord-checkpoint'
imported_graph = tf.compat.v1.train.import_meta_graph('/home/tony/ViMusic/models/remi/REMI-tempo-chord-checkpoint/model.meta')
# sess = tf.compat.v1.Session(config=config)
saver = tf.compat.v1.train.Saver()
with tf.compat.v1.Session() as sess:
    saver.restore(sess, '{}/model'.format(checkpoint))
    graph = tf.compat.v1.get_default_graph()
    # tf.compat.v1.train.write_graph(graph, './test/', 'remi_graph.pb', as_text=False)
    tf.compat.v1.summary.FileWriter('log/', graph=tf.compat.v1.get_default_graph()).close()
    # tf.compat.v1.saved_model.save(imported_graph, '~/test/')