# Logistic Regression

### Requirements
Required packages are listed in requirements.txt. Install them into your virtual environment with

    pip install -r requirements.txt
   
### Usage

    usage: evaluate.py [-h] [--dataset {iris,monk}]
                   [--features {0,1,2,3} [{0,1,2,3} ...]] [--exploration]
                   [--performance] [--decision-boundary] [--no-plot] [--safe]

    optional arguments:
      -h, --help            show this help message and exit
      --dataset {iris,monk}
                            the dataset to be used, either iris or monk
      --features {0,1,2,3} [{0,1,2,3} ...]
                            features used when dataset is iris
      --exploration         whether to plot the pairplot
      --performance         whether to plot performance measures
      --decision-boundary   whether to plot/save the decision boundary
      --no-plot             deactivate plotting for the decision boundary
      --safe                activate saving of decision boundary plot
      
# Model

The logistic regression model for binary classification is
single-layered and has only one output unit that models the probability
over the binary output by a Bernoulli distribution. The number of input
units is determined by the number of features in the dataset, but
augmented by one neuron of constant value 1 to represent the bias. A
*forward* pass through this model is hence performed by the following
equation:

<p align="center"><img src="./svgs/964a91ad2917985fb48db1d14f66e11b.svg" width=85.423965pt height=16.438356pt/></p>

That is, we take the dot product (sum of element-wise products) of weights (plus bias) <img src="./svgs/c6f31675281baa2569d8961577ecbf6b.svg" width=10.821921pt height=7.0776255pt/> and the augmented input <img src="./svgs/7073627e9999e583f5539cb4560a14d7.svg" width=9.3949845pt height=7.0776255pt/> and squeeze it into the range <img src="./svgs/d168c92829058f6af31167b13cce26f0.svg" width=36.529845pt height=24.6576pt/> using the sigmoid function <img src="./svgs/06bada42a49fa544331f5feb92c670dd.svg" width=9.982896pt height=7.0776255pt/>.

To update <img src="./svgs/c6f31675281baa2569d8961577ecbf6b.svg" width=10.821921pt height=7.0776255pt/>, we employ the Gradient Descent algorithm with batch size 1, i.e., Stochastic Gradient Descent (SGD). Thus, during training, at each step the model samples one instance from the dataset without replacement. It then performs the forward pass to make a prediction <img src="./svgs/d62f9c2bf8726d76e17edfaec186f464.svg" width=9.3474975pt height=14.611872pt/>. The objective of logistic regression is the maximization of the log likelihood of true labels <img src="./svgs/8dfa08d909b122145492276ec756f3fa.svg" width=8.6492175pt height=10.2739725pt/> given data <img src="./svgs/7073627e9999e583f5539cb4560a14d7.svg" width=9.3949845pt height=7.0776255pt/>
and parameters <img src="./svgs/c6f31675281baa2569d8961577ecbf6b.svg" width=10.821921pt height=7.0776255pt/>. We can maximize by minimizing its negation using Gradient Descent. The gradient of the logistic loss for weight <img src="./svgs/6ac91b4e7dd35551c6ea477deba5f82d.svg" width=5.663229pt height=10.8415065pt/> is given by

<p align="center"><img src="./svgs/4e687ade87553e995ccb1b5b1fdb6823.svg" width=140.91792pt height=16.438356pt/></p>

Based on the prediction <img src="./svgs/d62f9c2bf8726d76e17edfaec186f464.svg" width=9.3474975pt height=14.611872pt/> from the forward pass, the true label <img src="./svgs/8dfa08d909b122145492276ec756f3fa.svg" width=8.6492175pt height=10.2739725pt/> and the input feature <img src="./svgs/96de47a534893e2f93c9edceffaef3d1.svg" width=14.045889pt height=9.5433525pt/>, SGD then updates the weights using the delta rule

<p align="center"><img src="./svgs/159164f7edf5fbc96db92332a20278e1.svg" width=177.47895pt height=16.438356pt/></p>

The negation of <img src="./svgs/01115551f5e60fd9bf679e6d9def7437.svg" width=49.391265pt height=16.438356pt/> is necessary since we minimize in gradient descent, but want to maximize the log likelihood (and therefore minimize its negation). <img src="./svgs/6f532d874cff327b5508121b0a26c178.svg" width=8.751963pt height=10.2739725pt/> is the learning rate that controls the step size of the Gradient Descent. If steps are too small, learning will take too long. If the steps are too large, it can happen that the optimization does not converge to the minimum but oscillates around it.

Furthermore, a regularization term is added to the loss <img src="./svgs/4e23420b520032a25ea27771a93d6533.svg" width=34.303665pt height=16.438356pt/>, controlled by the weight decay rate <img src="./svgs/18f8eacfb4280d2c13c04e23edc6650d.svg" width=9.5890905pt height=11.415525pt/>:

<p align="center"><img src="./svgs/077404b64d095d2445485edda8a31db4.svg" width=47.201055pt height=33.81213pt/></p>

The partial derivative that needs to be added to the gradient of equation [\[eq:gradient\]](#eq:gradient) is

<p align="center"><img src="./svgs/497756bc01cc515dbe0fd399681b10f3.svg" width=29.860215pt height=13.881252pt/></p>

Since we do not want to prevent the bias from taking any necessary value, we only apply weight decay to all <img src="./svgs/7f7a18140b9af76bca6df8935c37b126.svg" width=14.883033pt height=9.5433525pt/> except for <img src="./svgs/d1a2f6d69afe0ced3bc7d32fef39f3ab.svg" width=16.784625pt height=9.5433525pt/>.

![Pairplot of the iris
dataset<span label="fig:explo"></span>](figures/explodata.png)

# Data

The logistic regression model described above is tested on two datasets:
the *iris* dataset and the *monk* dataset. The **iris dataset** consists
of three classes, with 50 data points each. Figure [1](#fig:explo)
visualizes the dataset in a pairplot. It becomes apparent from this plot
that the classes *setosa* and *versicolor* can be unambigously
distinguished by the use of any two features. We will leverage this fact
in order to make the above described model applicable to the iris
dataset by only taking the first 100 samples into account and hence
dropping the last class. The clear distinguishability of the remaining
classes allows us to experiment with only 2 input features that we can
plot along the learned decision boundary.

The **monk dataset** is an artificial dataset consisting of 2
classes. It can therefore be used out of the box for this model. There
are a total of 432 samples with 6 features, all in the form of
integers.

#### Preprocessing

As both *iris* and *monk* are presented in convenient format, not much
preprocessing is necessary. In fact, we only apply normalization of the
features by using z-score standardization given as

<p align="center"><img src="./svgs/c35e3aa92f84213584bd304c19825906.svg" width=77.288145pt height=31.98558pt/></p>

where <img src="./svgs/07617f9d8fe48b4a7b3f523d6730eef0.svg" width=9.90495pt height=14.15535pt/> is the mean and <img src="./svgs/8cda31ed38c6d59d14ebefa440099572.svg" width=9.982995pt height=14.15535pt/> the standard deviation of the
feature. For both datasets, the samples are randomly distributed over
train and test set in a 80/20 split.

# Results

![Development of train set and test set loss and accuracy over 30 epochs
on the iris
dataset.<span label="fig:perfiris"></span>](figures/performance.png)

## Iris Dataset

![Decision boundaries from Figure
[\[fig:decision-boundaries\]](#fig:decision-boundaries) but with the
test data
points.<span id="fig:decision-boundaries-test" label="fig:decision-boundaries-test">\[fig:decision-boundaries-test\]</span>](figures/db_0_1.png)
![Decision boundaries from Figure
[\[fig:decision-boundaries\]](#fig:decision-boundaries) but with the
test data
points.<span id="fig:decision-boundaries-test" label="fig:decision-boundaries-test">\[fig:decision-boundaries-test\]</span>](figures/db_0_2.png)
![Decision boundaries from Figure
[\[fig:decision-boundaries\]](#fig:decision-boundaries) but with the
test data
points.<span id="fig:decision-boundaries-test" label="fig:decision-boundaries-test">\[fig:decision-boundaries-test\]</span>](figures/db_0_3.png)
![Decision boundaries from Figure
[\[fig:decision-boundaries\]](#fig:decision-boundaries) but with the
test data
points.<span id="fig:decision-boundaries-test" label="fig:decision-boundaries-test">\[fig:decision-boundaries-test\]</span>](figures/db_1_2.png)
![Decision boundaries from Figure
[\[fig:decision-boundaries\]](#fig:decision-boundaries) but with the
test data
points.<span id="fig:decision-boundaries-test" label="fig:decision-boundaries-test">\[fig:decision-boundaries-test\]</span>](figures/db_1_3.png)
![Decision boundaries from Figure
[\[fig:decision-boundaries\]](#fig:decision-boundaries) but with the
test data
points.<span id="fig:decision-boundaries-test" label="fig:decision-boundaries-test">\[fig:decision-boundaries-test\]</span>](figures/db_2_3.png)

![Decision boundaries from Figure
[\[fig:decision-boundaries\]](#fig:decision-boundaries) but with the
test data
points.<span id="fig:decision-boundaries-test" label="fig:decision-boundaries-test">\[fig:decision-boundaries-test\]</span>](figures/db_0_1_test.png)
![Decision boundaries from Figure
[\[fig:decision-boundaries\]](#fig:decision-boundaries) but with the
test data
points.<span id="fig:decision-boundaries-test" label="fig:decision-boundaries-test">\[fig:decision-boundaries-test\]</span>](figures/db_0_2_test.png)
![Decision boundaries from Figure
[\[fig:decision-boundaries\]](#fig:decision-boundaries) but with the
test data
points.<span id="fig:decision-boundaries-test" label="fig:decision-boundaries-test">\[fig:decision-boundaries-test\]</span>](figures/db_0_3_test.png)
![Decision boundaries from Figure
[\[fig:decision-boundaries\]](#fig:decision-boundaries) but with the
test data
points.<span id="fig:decision-boundaries-test" label="fig:decision-boundaries-test">\[fig:decision-boundaries-test\]</span>](figures/db_1_2_test.png)
![Decision boundaries from Figure
[\[fig:decision-boundaries\]](#fig:decision-boundaries) but with the
test data
points.<span id="fig:decision-boundaries-test" label="fig:decision-boundaries-test">\[fig:decision-boundaries-test\]</span>](figures/db_1_3_test.png)
![Decision boundaries from Figure
[\[fig:decision-boundaries\]](#fig:decision-boundaries) but with the
test data
points.<span id="fig:decision-boundaries-test" label="fig:decision-boundaries-test">\[fig:decision-boundaries-test\]</span>](figures/db_2_3_test.png)

For experiments on the Iris dataset, the learning rate is set to
0.05 and the weight decay to 0.005. Figure [2](#fig:perfiris)
shows the development of loss and accuracy over <img src="./svgs/08f4ed92f27cec32cdd7a6ecd580f9e7.svg" width=16.438455pt height=21.18732pt/> epochs of
training. Figure [\[fig:decision-boundaries\]](#fig:decision-boundaries)
and Figure
[\[fig:decision-boundaries-test\]](#fig:decision-boundaries-test) show
the decision boundaries trained with different feature pairs along the
training and testing data points respectively.

As these figures show, the learned decision boundaries separate the two
clusters of points corresponding to two classes without errors. The
decision boundaries also work for the unseen test data. Note though,
that for example in the two bottom left plots of Figure
[\[fig:decision-boundaries\]](#fig:decision-boundaries) there are some
data points lying further away from the cluster. If these would have not
been part of the training set, then the decision boundary may have been
further tilted in a way minimizing the loss of the training data, but
misclassifying these "outliers" in the test set.

### Effect of Regularization

Table 1 shows results on the iris dataset (using features 0 and 2) after
200 epochs for three configurations: no regularization at all,
regularization with <img src="./svgs/792f9d027e08cf5bcb47332a94c10a94.svg" width=68.94987pt height=22.83138pt/> and regularization with
<img src="./svgs/3241b9e887b30bfedec9223399a548c0.svg" width=60.73056pt height=22.83138pt/>. Only the loss is reported as the accuracy stays at
<img src="./svgs/e065f2ebe0e614eff29483ac1b642605.svg" width=38.356395pt height=24.6576pt/> and does not allow for nuanced analysis. With a low
<img src="./svgs/792f9d027e08cf5bcb47332a94c10a94.svg" width=68.94987pt height=22.83138pt/> both the train and test loss decrease. With a higher
weight decay of <img src="./svgs/3241b9e887b30bfedec9223399a548c0.svg" width=60.73056pt height=22.83138pt/>, the test loss\[1\] further decreases
while the train loss increases. While the model is still very good for
the test data, we hence observe first signs of underfitting due to too
much regularization. These results were consistent using different
random seedings.

|                   | **Train** | **Test** |
| :---------------- | :-------- | :------- |
| No Regularization | 0.032     | 0.0173   |
| 0.005             | 0.0218    | 0.0079   |
| 0.05              | 0.3568    | 0        |

Loss after 200 epochs on the iris dataset using no regularization,
weight decay with rate <img src="./svgs/1f3e30a75633bef3b9280e9be2e1ce1d.svg" width=37.44312pt height=21.18732pt/> and rate
<img src="./svgs/09b35b77d506cef3840e129c2e29ed1f.svg" width=29.223975pt height=21.18732pt/>.<span label="tab:regularization"></span>

### Learning Rate Tuning

![Convergence (loss) over 100 epochs on the iris dataset for different
learning rates.<span label="fig:lr"></span>](figures/lr.png)

We can explore the effects of different learning rates <img src="./svgs/1d0496971a2775f4887d1df25cea4f7e.svg" width=8.752095pt height=14.15535pt/> by
plotting the convergence of the model when using them. Such a plot is
given in Figure [3](#fig:lr). It can be clearly observed, that lower
learning rates cause slower convergence, but for this simple problem do
not results in better convergence at later epochs. Given the very low
loss achieved after only few epochs when using higher learning rates
such as <img src="./svgs/358d4d0949e47523757b4bc797ab597e.svg" width=21.004665pt height=21.18732pt/> shows that the simplicity of the problem allows for
choosing such high learning rates.

## Monk Dataset

The monk data set turned out to be a more complicated problem and the
model did not manage to achieve <img src="./svgs/e065f2ebe0e614eff29483ac1b642605.svg" width=38.356395pt height=24.6576pt/> accuracy. Figure
[4](#fig:perfmonk) shows the development of loss and accuracy. As a
learning rate, <img src="./svgs/1f3e30a75633bef3b9280e9be2e1ce1d.svg" width=37.44312pt height=21.18732pt/> showed to produce good results based on
informal experiments. A weight decay actually showed to be harmful in
this experiment, which is why it is deactivated here. Table
[\[tab:monk\]](#tab:monk) also shows exact results on the monk dataset
after <img src="./svgs/88db9c6bd8c9a0b1527a1cedb8501c55.svg" width=24.657765pt height=21.18732pt/> epochs.

![Development of train set and test set loss and accuracy over 30 epochs
on the monk
dataset.<span label="fig:perfmonk"></span>](figures/performance_monk.png)

|              | **Train** | **Test** |
| :----------- | :-------- | :------- |
| **Accuracy** | 0.758     | 0.8      |
| **Loss**     | 0.457     | 0.36     |

Loss and Accuracy after 200 epochs on the monk dataset.
