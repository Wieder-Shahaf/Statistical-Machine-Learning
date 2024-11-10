<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>Machine Learning Techniques and Model Evaluation</h1>

<h2>Overview</h2>
<p>This project encompasses exercises focusing on k-Nearest Neighbors, Support Vector Machines, gradient-based optimization, cross-validation, and decision tree-based methods. Each exercise highlights specific machine learning techniques and model evaluation strategies:</p>
<ol>
    <li><strong>k-Nearest Neighbors (k-NN):</strong> Implementing and evaluating k-NN for multi-class classification with custom tie-breaking.</li>
    <li><strong>Support Vector Machines (SVM) and Perceptron:</strong> Implementing SVM using SGD, alongside training a Perceptron model.</li>
    <li><strong>Gradient Descent and Stochastic Gradient Descent (SGD):</strong> Optimizing convex functions with GD and applying SGD for SVM.</li>
    <li><strong>SVM Evaluation with Cross-Validation on MNIST-Fashion Dataset:</strong> Applying cross-validation to SVM models with various kernels and hyperparameters.</li>
    <li><strong>Decision Tree and Random Forest on Wine Quality Dataset:</strong> Evaluating Decision Trees and Random Forests for binary classification on wine quality data.</li>
</ol>

<h2>k-Nearest Neighbors (Exercise 1)</h2>
<p>This exercise involved implementing a multi-class k-Nearest Neighbors (k-NN) algorithm with a custom tie-breaking rule:</p>
<ul>
    <li>Defining the k-NN algorithm for classification with multiple classes.</li>
    <li>Implementing a tie-breaking rule to handle cases where multiple classes are equally close neighbors.</li>
</ul>
<p><strong>Challenge:</strong> Ensuring the algorithm correctly classified instances across classes and resolved ties accurately without using external libraries.</p>

<h2>Support Vector Machines (SVM) and Perceptron (Exercise 2)</h2>
<p>This exercise focused on implementing SVM using SGD and training a Perceptron model:</p>
<ul>
    <li>Using SGD to optimize a soft-margin SVM model.</li>
    <li>Training a Perceptron and analyzing its convergence on binary classification tasks.</li>
</ul>
<p><strong>Challenge:</strong> Balancing learning rate and initialization to achieve optimal performance for both models.</p>

<h2>Gradient Descent and Stochastic Gradient Descent (Exercise 3)</h2>
<p>This exercise required implementing gradient descent (GD) and stochastic gradient descent (SGD) to find minima of convex functions:</p>
<ul>
    <li>Defining a differentiable convex function and plotting it.</li>
    <li>Implementing its gradient and applying GD and SGD to find the function’s minimum.</li>
</ul>
<p><strong>Challenge:</strong> Adjusting the learning rate for GD and SGD to ensure stable convergence without oscillations.</p>

<h2>SVM Evaluation with Cross-Validation on MNIST-Fashion Dataset (Exercise 4)</h2>
<p>This exercise involved using cross-validation to evaluate SVM models on the MNIST-Fashion dataset with various kernels and parameters:</p>
<ul>
    <li>Loading and visualizing a subset of the dataset with labels.</li>
    <li>Implementing SVM models with linear, polynomial, and RBF kernels, then evaluating errors using cross-validation.</li>
    <li>Visualizing model performance through bar plots of training, validation, and test errors for each SVM model.</li>
</ul>
<p><strong>Best Model Selection:</strong> We compared the best model from cross-validation with the best-performing model on the test set to understand generalization.</p>

<h2>Decision Tree and Random Forest on Wine Quality Dataset (Exercise 5)</h2>
<p>This exercise applied Decision Trees and Random Forests to the wine quality dataset for binary classification:</p>

<h3>Data Preprocessing</h3>
<ul>
    <li>Loaded the <code>red-wine-quality.csv</code> dataset and converted it into a pandas DataFrame.</li>
    <li>Transformed the <code>quality</code> column to a binary label, where values &lt;= 5 were labeled as 0, and values &gt; 5 as 1.</li>
    <li>Split the data into features (<code>X</code>) and labels (<code>y</code>), then further into training and test sets with a 60-40 split.</li>
</ul>

<h3>Model Training and Evaluation</h3>
<ul>
    <li><strong>Decision Tree Classifier:</strong> Configured with <code>max_depth=12</code> and <code>random_state=0</code>. The model was trained on the training set, and accuracy was reported for both the training and test sets.</li>
    <li><strong>Random Forest Classifier:</strong> Configured with <code>n_estimators=100</code>, <code>max_depth=12</code>, and <code>random_state=0</code>. Accuracy was reported on both sets, and a plot of test accuracy was generated as a function of the number of trees (1 to 100).</li>
    <li><strong>Random Forest with No Feature Sampling:</strong> Reconfigured to consider all features at each split by setting <code>max_features=None</code>. The model was trained and tested as above, and a plot of test accuracy was generated as a function of the number of trees (1 to 100).</li>
</ul>

<h3>Analysis and Results</h3>
<ul>
    <li><strong>Comparison of Decision Tree and Random Forest:</strong> We analyzed whether the Random Forest classifier performed better than the Decision Tree and provided reasoning based on ensemble averaging and variance reduction in Random Forests.</li>
    <li><strong>Feature Sampling Effect:</strong> We compared the performance of the Random Forest with and without feature sampling, discussing the impact on accuracy due to the diversity of trees and potential overfitting.</li>
</ul>

<h3>Tree Splitting and Impurity Metrics (Theoretical)</h3>
<p>This section explored the concept of tree impurity metrics through hypothetical splits:</p>
<ul>
    <li><strong>Split Quality:</strong> Determining which of two splits better separated classes.</li>
    <li><strong>Impurity Metrics:</strong> Calculating misclassification rate, Gini index, and entropy for each split to evaluate which metric should be avoided for decision tree construction.</li>
</ul>

<h2>Conclusion</h2>
<p>This project demonstrated the application of machine learning algorithms including k-NN, SVM, gradient descent, cross-validation, and ensemble methods. Each exercise provided insights into optimizing model performance through parameter tuning, evaluation techniques, and metric analysis.</p>

</body>
</html>
