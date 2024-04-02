# ID3
ID3 stands for Iterative Dichotomiser 3 and is named such because the algorithm iteratively (repeatedly) dichotomizes(divides) features into two or more groups at each step. ID3 uses a top-down greedy approach to build a decision tree
ID3 uses Information Gain or just Gain to find the best feature. Information Gain calculates the reduction in the entropy and measures how well a given feature separates or classifies the target classes. The feature with the highest Information Gain is selected as the best one. In simple words, Entropy is the measure of disorder and the Entropy of a dataset is the measure of disorder in the target feature of the dataset.
ID3 Steps
•	Calculate the Information Gain of each feature.
•	Considering that all rows don’t belong to the same class, split the dataset S into subsets using the feature for which the Information Gain is maximum.
•	Make a decision tree node using the feature with the maximum Information gain.
•	If all rows belong to the same class, make the current node as a leaf node with the class as its label.
•	Repeat for the remaining features until we run out of all features, or the decision tree has all leaf nodes.
