# Feature_Selection
### Feature selection approaches and their implementation in Python. In feature seleciton we try to find the most consistent, non-redundant, and relevant features to use in a Machine Learning model.

#### There are different method for Feature selection. They are mentioned here, with an example:

#### Purpose of Feature selection:
* Reducing computentional costs
* Imporve the performance of model

#### Gained benefits from feature selection before applying ML model:
* simpler model, then easier to explain it.
* Avoid the curse of high dimensionality.
* Shorter training time, as we have lower feature dimension and also more precise subset of features. 
* Variance reduction

#### Feature selection methods:
Feature selection mostly has been done using following methods:

* Filter methods:
Selecting features based on statistics. They can be:
    - Univariate: each feature's affect on output studies individually.
    - Multivariate: evaluates the relevance of the features as a whole. 

* Wrapper methods:
They are not about selecting a feature or evaluating their affect on result, in wrapper methods the purpose is to select a set of features. Then the interactions between features would be detected too. (Boruta feature selection and Forward feature selection)

* Embeded methods:
Feature selection is a part of learning procedure, so classification and feature selection are performed simultaneously. (Random forest feature selection, decision tree feature selection, LASOO feature selection)

#### How to choose feature selection method:
* Numerical input, Numerical Output: (Regression)
    Using a correlation coefficient such as Pearson's (linear) or Spearmanc's (non-linear)
* Numerical input, Categorical output: (Classification)
    ANOVA (linear) or Kendall's (non-linear)
* Categorical input, Numerical output: (regression predictive modeling)
    ANOVA (linear) or Kendall's (non-linear) but in reverse.
* Categorical input, Categorical output: (classification predictive modeling)
    Chi-Squared test

#### Dimensionality Reduction Techniques:
* Percent missing values
* Amount of variation
* Pairwise correlation
* Multicolinearity
* Principle Component Analysis (PCA)
* Cluster Analysis
* Correlation (with target)
* Forwards selection
* Backward elimination (RFE)
* Stepwise selection
* LASSO
* Tree-based selection