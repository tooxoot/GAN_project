# 1 (Introduction)
Good Morning/Evening i'm Sven Vaupel currently studying software engineering here at this university and welcome to my talk about **Generative Adversarial Networks in text generation**.

# 2 (Agenda)
In this talk I will cover:
- `x` Machine Learning in generall
- What the **basic priciples** are and whow they are applied to **simplified examples**

- `x` Afterward I will introduce you to Generative Adversarial Networks or **GANs**
- Again giving you an Overview over the basics

- `x` And finally my talk will cover  how the **GAN model is applied to text generation**
- With special regards to an **architecture called seqGan**

# 3 (Machine Learning)
- `x` By now you probably guessed that this talk is about machine learning.
- But **what's the deal** with that?

# 4 (ML - Basic priciples)
- `x`
- Machine learning is all about
- `x`
-  The **creation of problem solving algorithms** without actually programming them.
- (For Example creating an algorithm that recognizes **hand written digits** is a **hard** Problem. But we can **easily** create one **using Machine Learing** technique.)
- `x`
- This creation can be boiled down to **optimizing mathematical functions to fit our tasks**.
- Regarding the so called **Supervised Learing** we need to know what our
- `x`
- **Inputs** look and what should
- `x`
- **come out** of the
- `x`
- **algorithm**
- `x`
- Which is then updated by **measuring its performance**.
- For this to work we also need lots of
- `x`
- Training data. Basically **pairs of input examples** an their **expected outputs**
- (Actually one of the two main reasons we are **currently making such progress in Machine Learing** is that we now have the amout of **data we need**. The other one has to do with **processing power**. Bac to the Topic)
- The actual **training procedure** then consist of gathering a **set of inputs**
- `x`
- `x`
- and putting the **through our unfinished Machine** to gain their **outputs**
- These **imperfect results** are then
- `x`
- compared to our **expectations**
- `x`
- the **outcome of this comparrison** is the used to **optimize** the algorithm.
- This **Training step** consisting of **Sampling, Calculating, Comparing and updating** is then repeated until the Machine's output nearly **matches the expectations**
- `x`

# 6 (ML - Basic Example)
- To visualize this process a bit more we'll have a look at an
- `x`
- Algorithm that determines if a given **image shows a cat**.
- Our **Training Data** will the consist of
- `x`
- **Images as inputs** obviously
- And a **value that indicates** if it's a cat or not.
- We will then give our algorithm a **basic shape** we can then optimize
- `x`
- During our **Training steps** we will then
- `x`
- Take some **Images**
- `x`
- **feed** them to the Machine
- **resulting in values** that resemble the **probability** of a picture showing a cat
- (In this case our algorithm is, with **different grades of certanty**, sure that these are all cats)
- These values are then **compared to the training results**
- `x`
- The **difference** between those values is the used to **update** the algorithm which is done via minimization of the underlying funktion.

# 8 (ML - Inner workings)
- So we saw how the training procedure is done, but not how the **ambigious basig shape** we have to define looks.
- When we **today** talk about Machine Learing we are most likely talking about
- `x`
- **Neural Networks**. And you might have seen some figure like this where each **neuron** as well as its **in and outgoing connections** are shown.
- Here we see a network consisting of
- `x`
- an **input layer**
- (Where in our example each **circle** would resemble one **pixelvalue** of the image)
- `x`
- and an **output layer**
- (Which in our example would just consist of **one value**, the **probability** that the input shows a cat )
- `x`
- and two **fully connected hidden layers**
- where each neuron resembles a **matrix operation with variable wheights and biases**
- This actually turns our "Machine Magic", as i called it on the slides, into a large **concatenation of mathematical functions**.
- As we **compare** the outputs to our expectations
- `x`
- this concatenation becomes quite handy.
- Now our **cost function** which we use for the comparrison is **parameterized by the current configuration** of the weights and biases in use.
- By calculating the cost **function's gradient** we are now able to calculate the **gradient of each neuron** throug an algorithm called **backpropagation**.
- This enables us to **follow along the different gradients to update each neuron's** weights and biases
- `x`
- thus **minimizing** the cost function for the **following Training steps**

# 9 (ML - Typical Tasks)
- Using these techniques it is possible to teach an algorithm to solve for example the following tasks:
- `x`
- **Classification**
- We saw this one. Having some Inputs which should be classified as one thing or the other.
- `x`
- **Regression**
- Here you try to predict data from a given set of information.
- Like predicting the future supply of cat pictures
- `x`
- And **Generation**
- Which means creating new samples from noise, according to existing references.
- So we can generate new cat imagery if our supply suddenly drops.

# 10 (GAN Title)
If you remembered the title you might have guessed already that this talk will feature the task of sample generation. And one Method to accomplish this is called Generative Adversarial Networks.

# 11 (GANs - Basic Priciples)
- When generation **new samples** our basig goal
- `x`
- is to be as close as possible to our **reference material**.
- (Here you see how you'd normally do so. Feed some **noise through your generator network**. )
- (**Compare** the result to the references via a **cost function. Update repeat**.)
- The **architecture** of Generative Adversarial Networks will now introduce a **second neural network called discriminator**.
- `x`
- And while the generator tries to **produce fake samples**
- `x`
- The discriminator will try to **seperate fakes from originals**.
- **Both** neural networks will be **trained with this information**
- `x`
- And while the discriminator is trained to **correctly identify original samples**.
- `x`
- The Generator is trained to **produce fake samples** which are actually **identified as originals by the discriminator**
- `x`
- (You can imagine the generator being an **art forger** trying to forge **expensive picassos** while the discriminator, some kind of **art experts** is trying to **protect the market**. Both the generator and the discriminator will get better at their work.)

# 14 (GANs - Simple Example)
- To **visualize this dynamic** we will go through the training of a simple example i trained myself
- `x`
- The references are **simple points drawn from a certain rectangle**.
- `x`
- The Generator had **two** fully connected layers of **10 neurons**
- `x`
- While the discriminator had **3 of those layers**.
- This together with the **output layers made over 700 weights and biases** to adjust.
- Each training steps took **500 samples** from the generator as well as the reference set.
- `x`
- Which you can see in this gaphic.
- This whole Set of **1000 points** was then put through the discriminator
- `x`
- As you can see at the **beginning** the both networks are quite **stupid**.
- So we train them.
- (Here Generator was **trained a bit less** to prevent overfitting the **discriminator's current configuration**)
- So after some cycles
- `x`
- the Generated distribution **will move towards the reference** data will
- `x`
- But our discriminator will also be **better at cutting** out the fakes
- `x`
- As this **process continues** you can see the discriminator getting quite **creative with its boudries**.
- until the **generator gets such good** that the discriminator just **cuts the distribution in half** because there is no way of **telling them appart**.
- At this stage the training is done and you have a fine generation algorithm.

# 16 (GANs - Overview)
- So why GANs? To make it short:
- `x`
- They are **based on Neural Networks** and thus accessible to our beloved **gradient based optimization**.
- Some of the **previously used methods were not**
- `x`
- The **major drawback** is that it's actually quite **tidious to train** them.
- The **weights' and biases' inital values** as well as the chosen **hyperparameters**,
-  like the **sample size** or the **ratio of training steps** between G and D,
-  are playing a huge role when it comes to the **training successes**.
- Luckily **pretraining the generaotr by direct comparrison** insted of using the discriminator immediatly might help to **stabilize the outcomes**.
- `x`
- When it comes to usage, the **success of GANs** in different Machine Learing **domains justify their use**.
- For example here you see a case of image translation.
- Horse to zebra translation to be precise.


# 17 (Text Generation using GANs)
Ok Now that we have the Background check out of the way we'll have a quick look on how text generation with Gans can work.

# 18 (TG - The Problem)

>> TODO complications one down

- As seen before our goal is to produce samples close to our reference data wich in this case means
- `x`
- Generating Text that resembles natural language.
- (Basically meaning that we want to close the gap between human written and machine produced text.)
- On the way to achieving this goal there are some difficulties
- `x`
- One beeing the inherent structure of text, like semantics and syntax.
- which Luckily are learnable features.
-`x`
- Then we have a very discrete Problem. Meaning that we have a limited vocabulary resulting in a limited number of possible sentences of a certain length.
- This is actualy a problem for some generation methods. but
- `x`
- Generative Adversarial Networks can handle that
- `x`
- The biggest Problem is the sequencial structure of natural language.
- If you want to continue this sentence for examples
- `x`
- 'The quick fox'
- A correct candidate would be flies so you can go on with
- `x`
- 'over the sea'
- or
- `x`
- 'next to me'
- But these Senteces are kind of nonsensical
- `x`
- Over the plain would be better but is kind of metaphorical.
- in conclusion 'flies' is correct but not a good candidate to continue with.
- `x`
- 'runs' is a better one. Leading to finished samples like
- `x`
- 'The quick fox runs down the hill'
- To evaluate what word to choose next we thus have to evaluate both the current content and possible outcomes.

# 19 (SeqGan - Model)

>> .1 and .2 directly on slide

- One GAN architecture that tries to tackle this evaluation is the so called SeqGAN model standing for 'Sequencial GAN'.
- `x`
- Taking the current sentence
- `x`
- and choosing the next word
- `x`
- gets really out of hand if you want to take ALL possible sentences into account. The Tree just expands it's a mess.
- What SeqGAN does instead
- `x`
- Is applying a method called monte carlo tree search
- `x`
- where just follow a certain number of paths for each cnadidate.
- `x`
- And then evaluate them by accumulating the rewards given to the final sentences.
- `x`
- The the sentences are generated by our Generator
- `x`
- and rewarded by the discriminator.
- Thus the picked candidate for the next word will be the one which has the highest probability of producing a sentece that fools the distcriminator.
  
