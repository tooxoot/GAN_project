## Machine Learning

- Creating an algorithm that solves a given problem without being programmed to.
- Optimizing mathematical functions regarding a given measurement

[Graphics 99: Input => Machine magic => output]

- Will have to define
  - the basic form of our function.
  - the performance measurment

- Problems:
  - Classification [Graphics 98]
  - Regression [Graphics 98]
  - Synthesis [Graphics 98] 
***
### Notes:

**Gerneral:**
As you probably recognized by now this talk is about machine learning.
But what do we mean when talking about this topic?
Generally speaking we want a program to improve its performance on a specific task, based on the experience it gained by working on it.
This will result in the creation of an algorithm solving a given problem without being specifically programmed to do so. [#]
How does this work you might ask?
Well Math! [#]

**Animation 1:**
When approaching problems with machine learning techniques we essentially optimizing the output [#] of complicated mathematical functions [#] applied on a given set of data [#] regarding to some performance  measurement [#].

**Animation 2:**
[#] In cases of supervised learing we will have a given set of training data consisting of inputs and their assigned results.
[#] The inputs are then fed to our algorithm wich will calculate some result [#].
We compare these calculations to the targeted Results from our training Data using a predefined measuremnt [#].
Then use this measuremnt to update out function [#].

**Animation 3:**
This explanation was kind of abstract! Let's imagine we want to create a machine that looks at an image [#] and tells us if it shows a cat or not [#].
(This task would be quite difficult to program by hand!)
Our training data would then consist of images [#] and their assigned labels.
By initializing a framework for our function [#] and feeding the Images through it [#], we would then get a more or less accurate labeling.
Using the actual labels from our training data we are then able to calculate a measurment [#] to feed back to our function, updating its parameters [#].
Repeating this cycle enough [#] times the function's parameters will approach an optimal state where fed images will be correctly classified as cats an non-cats.

***Conclusion:**

So in conclusion by defining th basic form of our function, what input to take and what output to calculate, in addition to defining a resonable performance measure we can tackle a variety of problems including:

Classification, where we assign labels to a dataset like we have seen in the example.

Regression, where we try to predict data based on given facts.

And Synthesis where we want to generate new examples of a given kind.
