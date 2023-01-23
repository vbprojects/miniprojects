# miniprojects
A collection of personal miniprojects, feel free to contribute.

## adjusted sigmoid optimization

Hard to explain, might right an article about it. It's what I refer too as surrogate objective.

### TODO

- More experiments

## Hyper Ellipsoid Classifier

Train ellipsoid classifier, I think it might make sense in latent space of a model. Also, good at finding clusters at least for moons data set.

### TODO

- Implement in higher dimensions
- classifier can currently only make positive predictions, probably because I haven't made radius an optimizable parameter.
- Try with different objective functions, ie. surrogate.

## ForwardForward

Implementation of Forward Forward algorithm described in https://www.cs.toronto.edu/~hinton/FFA13.pdf

### TODO

- ~~Figure out more accurate representation of algorithm, unsure if the program is actually implementing it despite promising preliminary results.~~

Implemented, need negative images to go further

## InterestingDistributions

A collection of probability distributions that I've come across.

### TODO

- ~~Figure out distribution to use as prior for probabilistic convex/non-convex hull through Hidden Markov Model.~~

Dead end

## Edge Detection

Currently, just a sobel edge detector

### TODO

- ~~Derive emission probabilities for a pixel being an edge given various filters.~~

Dead end, not interested enough

## kaggles

Kaggle notebooks

## kernelnetworks

Feature engineering through repeated kernel method application. For example, applying rbf on a rbf kernel. Produces interesting regressors and classifiers.

### TODO

- Figure out how to more accurately select subset of points for classifier, currently using kmeans++ initialization
- Figure out how to use more than 2 applications of rbf 
- Combat Runges Phenomon for extrapolation

## minvocab

Looking at classifiers that output binary labels. Each node corresponding to a 1 or 0 through a modification of the sigmoid function. 

### TODO

- If assigned binary labels MNIST labels take up 4 bits. 4 bits can represent 16 different states. Is there a way to boost classifier performance through mapping labels to more than one binary label? How do you train a machine that does that?

## misc articles

Title,

## OLS_Article

Notebooks concerning the derivation and optimization of Ordinary Least Squares problems.

### TODO

- ~~Get ready to put on medium or something.~~
DONE
Finished

## repeated_inner_outer_products

Title

### TODO

- ~~Maybe some kind of generative art?~~

Not enough there

## RotLogRegr

Logistic regression where boundary is determined through rotation.

### TODO

- How do you extrapolate this to higher dimensions?

## SVM_ARTICLE

Article on SVMS

### TODO

- ~~Implement SVC from scratch~~
- From scratch with Pegasos and not from scratch with JuMP, write content for article


