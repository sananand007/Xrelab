# Xrelab
My Work on Computer Vision and Deep Learning at Xrelab Sandiego

+ Lecture-1: Image classification of MNIST dataset
  + Using KNN
  + Using SVM

```Python
X=tf.placeholder(tf.float32, [None, 10304]) #gray scale so 1 value for pixel
W=tf.Variable(tf.zeros([10304, 40]))
b=tf.Variable(tf.zeros([40]))

#model
Y=tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 10304]), W) + b)

#placeholder for known labels
Y_=tf.placeholder(tf.float32, [None, 40]) #one hot encoded

#loss function
cross_entropy = -tf.reduce_sum(Y_*tf.log(Y))

#correct answers
is_correct=tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy=tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer=tf.train.GradientDescentOptimizer(0.003)
train_step=optimizer.minimize(cross_entropy)


# Matplotlib visualization
allweights=tf.reshape(W, [-1])
allbiases=tf.reshape(W,[-1])

# Run TF session
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)


train_accuracy_=[]
train_loss=[]
test_accuracy_=[]
test_loss=[]


batch_size=40
# Total number of images we have = 240 (40x6)
#Train the model 100 images at a time by the function below

def training_Step(i, update_test_data, update_train_data):
        # input batch genration
        batch_X, batch_Y = reshaped_train_set[:],reshaped_train_labels[:]

        # Backpropagation training step
        train_data = {X:batch_X, Y_:batch_Y}
        sess.run(train_step, feed_dict=train_data)

        #Compute training values for visualization
        if update_train_data:
            a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
            #a, c, im, w, b = sess.run([accuracy, cross_entropy, I, allweights, allbiases], feed_dict=train_data)
            #datavis.append_training_curves_data(i, a, c)
            #datavis.append_data_histograms(i, w, b)
            #datavis.update_image1(im)
            if i%100==0:
                train_accuracy_.append(a*100.0)
                train_loss.append(c)
                print("Iteration: {}, Train Acuuracy: {}, Train Loss: {}".format(i, a*100, c))

        #On the test data values for visualization
        if update_test_data:
            test_data={X:reshaped_test_set, Y_:reshaped_test_labels}
            a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
            #a, c, im = sess.run([accuracy, cross_entropy, It], feed_dict=test_data)
            #datavis.append_test_curves_data(i, a, c)
            #datavis.update_image1(im)
            if i%100==0:
                test_accuracy_.append(a*100.0)
                test_loss.append(c)
                print("Iterationg: {}, Test Acuuracy: {}, Test Loss: {}".format(i, a*100, c))

for i in range(1000+1): training_Step(i,True, True)

```
