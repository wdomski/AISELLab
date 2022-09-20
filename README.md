# Laboratory outline

## Lab -- scripting

Introduction to Python script language.

The goal of this laboratory class is to familiarize students 
with Python scripting language. 

In the course of this class a set of individual Python related
tasks will be handed out.

## Lab -- linear regression

Linear regression and gradient descent based methods.

### Linear regression

During class you will be given individual weight vector 
and bias coefficient for the linear model.
E.g.

```
w = np.array([1, 2, 3])
b = 4
```

Linear model is given as 

```
y = np.dot(w, x) + b
```

Generate 1000 random samples using np.random.rand().
The dimension of generated matrix containing samples 
should match the dimension of weight vector, thus
it should be 1000x3.

Extend samples vector with a column of ones 
to take into account bias.

Implement linear regression.

Compare estimated weight vector (extended with bias) 
with the original vector.

### Gradient descent based methods

Use previously generated samples, and the 
given weight vector and bias.

Implement:
- batch gradient descent, BGD,
- stochastic gradient descent, SGD,
- mini-batch gradient descent, MBGD.

Gradient descent based methods are based on two aspect:
calculation of gradients and iterative approach.

Prepare gradient calculation formula.

#### Batch gradient descent

At each iteration all samples are used to 
calculate gradient and then adjust estimated weights.

```
w_est = w_est - learning_rate * gradient
```
where learning_rate is scalar.

#### Stochastic gradient descent

With SGD a concept of epoch is introduced.
At each epoch an another iteration is taking place.
However, at each iteration only a single 
sample should be used to estimate weights. 
Each epoch is finished when the algorithm iterates 
through all samples.

Attention!

Iterating over set of samples should be random.

#### Mini-batch gradient descent

MBGD is similar to stochastic gradient descent.
However, instead of takin a single sample from 
the data set a subset of samples is taken.
Therefore, epoch finishes when the data set is emptied.

Attention!

Iterating over set of samples should be random.
It means that k samples should be drawn randomly from 
provided data set.

## Lab -- classification with ML

Classification of data based on Machine learning algorithms.

Students will be given or asked to download 
a complete dataset containing
features and labels. The goal of this class 
is to train classifier capable of assigning a class (predicted label)
given a vector of features.

Students will be given two different algorithms on which the solution 
should be based. 

Considered algorithms (but not limited to) are :
- SVM (Supported Vector Machine),
- KNN (K Nearest Neighbours),
- Decision tree, Random forest.

## Lab -- classification with DL

Classification of data based on Deep learning algorithms 
such as fully connected (dense) layers and CNNs.

Students will be given or asked to download 
a complete data set containing
features and labels. The goal of this class 
is to train classifier capable of assigning a class (predicted label)
given a vector of features.

The goal of this laboratory is to familiarize 
with dense layers and CNNs.
At the end of te classes comparison between these two 
methods should be presented.
A confusion matrix should be presented.

## Lab -- dataset augmentation

Sometimes it happens that the number of provided data samples 
is limited. Then, the original dataset should be extended.

During this laboratory class you will be given 
a limited dataset or asked to download one. 
Using data augmentation it should be extended.
Afterwards, this new dataset should be used to train a 
classifier.

At the end of this class student should present 
how well the trained model performs and if asked to 
comparison between model trained on full data set should be 
presented.

## Lab -- RNN

Recurrent neural networks are used to predict 
next, most probable, element in time series.

You will be given a dataset or asked to download one.
Based on this dataset you will need to prepare it 
for training. 
You will be given individual number of samples considered for 
learning process.

The goal is to create RNN based on LSTM 
(Long-Short Term Memory) and/or GRU (Gated Recurrent Unit).
You will be asked to present statistics how well 
trained network performed and also 
compare two different implementations between themselves.

## Lab -- mini project

Each of students will be given an individual taks. 

Examples of tasks

### Symbol detection

Consider following problem. You need to detect from image 
if presented shape belongs to one of the classes: 
circle, cross, square, moon (in eclipse).

You need to create dataset of example images.
All of them should have, small, identical size 
with white background and black drawing representing given 
shape.
Create 10 samples for each class.
Use data augmentation to create 1000 samples of each class.
Propose CNN architecture and perform training.
Present confusion matrix and other statistics.

Use web-based solution to deploy your model 
and allow user to upload an image offering as a 
result predicted class.

Extend interface to allow user to define 
his/her own classes. Perform training 
and deploy solution with web based approach.

To derive web-based approach Django can be used 
as a framework.

### Review classification

Create a dataset using web scrapping approach.

Train neural network to classify review based 
on words used as positive or negative one.

Prepare web-based solution to allow user 
to provide snippet of review and 
get predicted class.

Extend interface to add additional data samples to 
the data set, retrain model and deploy.
