# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:58:07 2019

신경망 구성 with tensorflow
"""

global_step = tf.Variable(0,trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2,10], -1., -1.), name='W1')
    L1 = tf.nn.relu(tf.matmul(X,W1))
    
    tf.summary.histogram("X", X)
    tf.summary.histogram("Weights", W1)
    
with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.randomm_uniform([10,20], -1., 1.), name='W2')
    L2 = tf.nn.relu(tf.matmul(L1, W2))
    
    tf.summary.histogram("Weights", W2)
    
with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_uniform([20,3], -1., 1.), name='W3')
    model = tf.matmul(L2, W3)
    
    tf.summary.histogram('Weights', W3)
    tf.summary.histogram('Model', model)
    
with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op= optimizer.optimize(cost, global_step=global_step)
    
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else: 
    sess.run(tf.global_variables_initializer())
    
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)
    

#-> tensorboard 띄움

for step in range(100): # epoch 직접 for loop으로 입력; 텐플 어려운 이유
    sess.run(train_op, feed_dict={X: x_data, Y:y_data})
    print('Step: %d ' % sess.run(global_step), 
          'Cost: %.3f' % sess.run(cost, feed_dict={X: x_Data, Y:y_data}))
    
    summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
    writer.add_summary(summary, global_step = sess.run(global_step))
    
    saver.save(Sess, './model/dnn.ckpt', global_step = global_step)
    
    #tensorboard로 결과 확인