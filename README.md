<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>Machine Learning Techniques and Model Evaluation</h1>

<h2>Overview</h2>
<p>This project includes exercises focusing on gradient-based optimization, classification models, cross-validation, and model tuning techniques. Each exercise demonstrates specific machine learning techniques and methods for model evaluation:</p>
<ol>
    <li><strong>Gradient Descent Optimization:</strong> Implementing and applying gradient descent on a simple convex function.</li>
    <li><strong>Soft-Margin SVM with Stochastic Gradient Descent:</strong> Training a soft-margin SVM model using SGD.</li>
    <li><strong>SVM Model Selection with k-Fold Cross-Validation:</strong> Performing cross-validation for SVM model selection and hyperparameter tuning.</li>
    <li><strong>SVM Evaluation on MNIST-Fashion Dataset:</strong> Applying cross-validation to SVM models with various kernels on the MNIST-Fashion dataset.</li>
    <li><strong>Decision Tree and Random Forest on Wine Quality Dataset:</strong> Evaluating Decision Trees and Random Forests for binary classification on wine quality data.</li>
</ol>

<h2>Gradient Descent Optimization (Exercise 1)</h2>
<p>This exercise involved implementing gradient descent (GD) on a convex quadratic function, focusing on the following steps:</p>
<ul>
    <li>Defining a differentiable convex function and plotting it.</li>
    <li>Implementing its gradient and applying GD to find the function’s minimum.</li>
</ul>
<p><strong>Challenge:</strong> Choosing an appropriate learning rate to ensure convergence without oscillation or divergence.</p>

<h2>Soft-Margin SVM with Stochastic Gradient Descent (Exercise 2)</h2>
<p>This exercise focused on implementing SGD for optimizing a soft-margin SVM model:</p>
<ul>
    <li>Implementing hinge loss with regularization for SVM.</li>
    <li>Using SGD updates to adjust model parameters for convergence.</li>
</ul>
<p><strong>Challenge:</strong> Tuning the learning rate and initialization to balance between convergence speed and model accuracy.</p>

<h2>SVM Model Selection with k-Fold Cross-Validation (Exercise 3)</h2>
<p>This exercise required a custom k-fold cross-validation function to facilitate SVM model selection and tuning:</p>
<ul>
    <li>Implementing a k-fold CV function without external libraries.</li>
    <li>Using this function to evaluate SVM models with varying regularization strengths.</li>
</ul>
<p><strong>Challenge:</strong> Finding the optimal regularization value to avoid overfitting while maximizing generalization.</p>

<h2>SVM Evaluation on MNIST-Fashion Dataset (Exercise 4)</h2>
<p>This exercise involved applying cross-validation on the MNIST-Fashion dataset using different SVM kernels and parameters:</p>
<ul>
    <li>Loading and visualizing a subset of the dataset with labels.</li>
    <li>Implementing SVM models with linear, polynomial, and RBF kernels and evaluating errors using cross-validation.</li>
    <li>Visualizing model performance through bar plots of training, validation, and test errors for each SVM model.</li>
</ul>
<p><strong>Best Model Selection:</strong> A comparison between the best model from CV and the one performing best on the test set provided insights into model generalizability.</p>

<h2>Decision Tree and Random Forest on Wine Quality Dataset (Exercise 5)</h2>
<p>This exercise used the wine quality dataset to evaluate Decision Trees and Random Forests for binary classification:</p>

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
    <li><strong>Comparison of Decision Tree and Random Forest:</strong> 
    We analyzed whether the Random Forest classifier performed better than the Decision Tree and provided reasoning based on ensemble averaging and variance reduction in Random Forests.</li>
    <li><strong>Feature Sampling Effect:</strong> 
    We compared the performance of the Random Forest with and without feature sampling, discussing the impact on accuracy due to the diversity of trees and potential overfitting.</li>
</ul>

<h3>Tree Splitting and Impurity Metrics (Theoretical)</h3>
<p>This section explored the concept of tree impurity metrics through hypothetical splits:</p>
<ul>
    <li><strong>Split Quality:</strong> Determining which of two splits better separated classes.</li>
    <li><strong>Impurity Metrics:</strong> Calculating misclassification rate, Gini index, and entropy for each split to evaluate which metric should be avoided for decision tree construction.</li>
</ul>

<h2>Conclusion</h2>
<p>This project demonstrated the application of gradient descent optimization, SVM model tuning with cross-validation, and classification with ensemble methods. Each exercise highlighted key considerations for achieving optimal performance through parameter tuning, model selection, and evaluation metrics.</p>

</body>
</html>
