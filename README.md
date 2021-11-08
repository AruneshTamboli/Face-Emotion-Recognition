# Live Class Monitoring System(Face Emotion Recognition)

![ExpNet_teaser_v2](https://user-images.githubusercontent.com/88345564/140759432-c85f97df-fd9b-4fdd-88a5-d97ca1fb4072.jpg)


# Problem Statement:
The Indian education landscape has been undergoing rapid changes for the past 10 years owing to the advancement of web-based learning services, specifically, eLearning platforms.

Global E-learning is estimated to witness an 8X over the next 5 years to reach USD 2B in 2021. India is expected to grow with a CAGR of 44% crossing the 10M users mark in 2021. Although the market is growing on a rapid scale, there are major challenges associated with digital learning when compared with brick and mortar classrooms. One of many challenges is how to ensure quality learning for students. Digital platforms might overpower physical classrooms in terms of content quality but when it comes to understanding whether students are able to grasp the content in a live class scenario is yet an open-end challenge. In a physical classroom during a lecturing teacher can see the faces and assess the emotion of the class and tune their lecture accordingly, whether he is going fast or slow. He can identify students who need special attention.

Digital classrooms are conducted via video telephony software program (ex-Zoom) where it’s not possible for medium scale class (25-50) to see all students and access the mood. Because of this drawback, students are not focusing on content due to lack of surveillance.

While digital platforms have limitations in terms of physical surveillance but it comes with the power of data and machines which can work for you. It provides data in the form of video, audio, and texts which can be analyzed using deep learning algorithms.

Deep learning backed system not only solves the surveillance issue, but it also removes the human bias from the system, and all information is no longer in the teacher’s brain rather translated in numbers that can be analyzed and tracked.

I will solve the above-mentioned challenge by applying deep learning algorithms to live video data. The solution to this problem is by recognizing facial emotions.

# Dataset Information:
I have built a deep learning model which detects the real time emotions of students through a webcam so that teachers can understand if students are able to grasp the topic according to students' expressions or emotions and then deploy the model. The model is trained on the FER-2013 dataset .This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised. Here is the dataset link:- https://www.kaggle.com/msambare/fer2013

# Model Creation

# 1) Using DeepFace

DeepFace is a deep learning facial recognition system created by a research group at Facebook. It identifies human faces in digital images. The program employs a nine-layer neural network with over 120 million connection weights and was trained on four million images uploaded by Facebook users.The Facebook Research team has stated that the DeepFace method reaches an accuracy of 97.35% ± 0.25% on Labeled Faces in the Wild (LFW) data set where human beings have 97.53%. This means that DeepFace is sometimes more successful than the human beings. Andrew-Ng-anger
![3](https://user-images.githubusercontent.com/88345564/140759597-80363460-fd6e-4cfc-90fa-f854aa2fa0fa.PNG)

• The actual emotion of Andew Ng is sad Face or using DeepFace I found the prediction is correct as well as his age limit.

# 2) Xception
Xception architecture is a linear stack of depth wise separable convolution layers with residual connections. This makes the architecture very easy to define and modify; it takes only 30 to 40 lines of code using a high level library such as Keras or Tensorflow not unlike an architecture such as VGG-16, but rather un- like architectures such as Inception V2 or V3 which are far more complex to define. An open-source implementation of Xception using Keras and Tensorflow is provided as part of the Keras Applications module2, under the MIT license. We used Adam as our optimizer after training for 70 epochs using Adam and a batch size of 785, we achieved 64% accuracy on the test set.
![xception1](https://user-images.githubusercontent.com/88345564/140759623-7f17bc5d-a5eb-45b2-b64a-3ec8c9c56063.jpg)



The above image shows the final infrastructure of the Xception model. A fully connected neural layer that contains residual depth wise separable convolution where each convolution followed by batch normalization and Relu activation function. The last layer applies a global average pooling and softmax activation function to produce prediction.

# 3)DeXpression model-

We propose a convolutional neural network (CNN) architecture for facial expression recognition. The proposed architecture is independent of any hand-crafted feature extraction and performs better than the earlier proposed convolutional neural network based approaches. We visualize the automatically extracted features which have been learned by the network in order to provide a better understanding. we achieved 63 % accuracy on the test set

# 4) Custom Deep CNN
![Capture](https://user-images.githubusercontent.com/88345564/140759670-28391c3f-04c0-4f24-915e-037c416a7329.PNG)

A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other.


We designed the CNN through which we passed our features to train the model and eventually test it using the test features. To construct CNN we have used a combination of several different functions such as sequential, Conv (2D), Batch Normalization, Maxpooling2D, Relu activation function, Dropout, Dense and finally softmax function.

We used Adam as our optimizer after training for 70 epochs using Adam with minimum learning rate 0.00001 and a batch size of 785, we achieved 69 % accuracy on the test set and 74% as train accuracy.
One drawback of the system is the some Disgust faces are showing Neutral .Because less no. of disgust faces are given to train .This may be the reason.
I thought it was a good score should improve the score.
Thus I decided that I will deploy the model.


# 5) Using Transfer Learning Resnet50

![Capture1](https://user-images.githubusercontent.com/88345564/140759731-b500b1bf-d93c-4006-8b07-d0adf2b02787.PNG)

Since the FER2013 dataset is quite small and unbalanced, we found that utilizing transfer learning significantly boosted the accuracy of our model. ResNet50 is the first pre-trained model we explored. ResNet50 is a deep residual network with 50 layers. It is defined in Keras with 175 layers. We replaced the original output layer with one FC layer of size 1000 and a softmax output layer of 7 emotion classes. We used Adam as our optimizer after training for 50 epochs using Adam and a batch size of 785, we achieved 63.11% accuracy on the test set and 67% on the train set. There is much less over-fitting. We have taken epochs as 50. Once the threshold is achieved by the model and we further tried to train our model, then it provided unexpected results and its accuracy also decreased. After that, increasing the epoch would also not help. Hence, epochs play a very important role in deciding the accuracy of the model, and its value can be decided through trial and error.

# Loss & Accuracy Plot

![2](https://user-images.githubusercontent.com/88345564/140759760-14ee0b20-bc4a-44d8-b7f0-ad0aaa246ebe.PNG)

# Realtime Local Video Face Detection
We created patterns for detecting and predicting single faces and as well as multiple faces using OpenCV videocapture in local. For Webapp , OpenCV can’t be used. Thus, using Streamlit-Webrtc for front-end application.

Deployment of Streamlit WebApp in Heroku and Streamlit

We deployed the app in Heroku if you saw in the starting section of github repo, you can see the all the requirement files are there for creating an app on Heroku of name “face-emotion-recognition-ofg”.

But due to high slug size the buffering takes time so we have ran our app working on local and it ran properly and app is also fine also we’ve included video on github repo.

streamlit link:-https://share.streamlit.io/aruneshtamboli/face-emotion-recognition/main/aruneshapp.py

Heroku Link:- https://proponent-ds.herokuapp.com/

# Conclusion:

We build the WebApp using streamlit and deployed in Heroku and Streamlit Sharing.

The model which was created by custom CNN model gave training accuracy of 77% and test accuracy of 69%.

Codes which we deployed are in Github Repository.

we did 11 times experiment with diffrent test train split and data normalization beacause we know that for NN large train data set gives better result of model output

It was such an amazing and interesting project. We learnt a lot from this.

Some Real Life Experience from thing amazing project

Understand the deep concept of project,and feel yes future is with this technique

Don't afraid to faliure beacause we invest 80 hr for colab due to limited Gpu for diffrent experimnet

Never Loose Hope.

Never Give Up!

Have some patience good things happen.

Try new things and execute your idea.
