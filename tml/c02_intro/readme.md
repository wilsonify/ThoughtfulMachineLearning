# Quick Introduction to Machine Learning

Welcome to the Quick Introduction to Machine Learning repository! 

a guide to understanding the fundamentals of machine learning and provides a general framework for thinking about machine learning algorithms.

## Algorithm Matrix 

| Algorithm                      | Learning type            | Class                              | Restriction bias                                                              | Preference bias                                                                                      |
| ------------------------------ | ------------------------ | ---------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| K-Nearest Neighbors            | Supervised               | Instance based                     | Generally speaking                                                            | KNN is good for measuring distance-based approximations; it suffers from the curse of dimensionality |
| Naive Bayes                    | Supervised               | Probabilistic                      | Works on problems where the inputs are independent from each other            | Prefers problems where the probability will always be greater than zero for each class               |
| Decision Trees/ Random Forests | Supervised               | Tree                               | Becomes less useful on problems with low covariance                           | Prefers problems with categorical data                                                               |
| Support Vector Machines        | Supervised               | Decision boundary                  | Works where there is a definite distinction between two classifications       | Prefers binary classification problems                                                               |
| Neural Networks                | Supervised               | Nonlinear functional approximation | Little restriction bias                                                       | Prefers binary inputs                                                                                |
| Hidden Markov Models           | Supervised/ Unsupervised | Markovian                          | Generally works well for system information where the Markov assumption holds | Prefers time-series data and memoryless information                                                  |
| Clustering                     | Unsupervised             | Clustering                         | No restriction                                                                | Prefers data that is in groupings given some form of distance (Euclidean                             |
| Feature Selection              | Unsupervised             | Matrix factorization               | No restrictions                                                               | Depending on algorithm can prefer data with high mutual information                                  |
| Feature Transformation         | Unsupervised             | Matrix factorization               | Must be a nondegenerate matrix                                                | Will work much better on matrices that don’t have inversion issues                                   |
| Bagging                        | Meta-heuristic           | Meta-heuristic                     | Will work on just about anything                                              | Prefers data that isn’t highly variable                                                              |