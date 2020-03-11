# Import standard libresies for data manipulation and plot utilities
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from os.path import join, dirname
import sys

# Import scikit learn utilities
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False,
                          title=None, cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]

    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im)
    # We want to show all ticks...
    plt.gca().set(xticks=np.arange(cm.shape[1]),
                  yticks=np.arange(cm.shape[0]),
                  xticklabels=classes, yticklabels=classes,
                  title=title,
                  ylabel='True label',
                  xlabel='Predicted label')

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.gca().set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    plt.gca().set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
    plt.grid()
    plt.tight_layout()

    return None


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.1, 1, 5)):
    # Details: http://scikit-learn.org/stable/auto_examples/model_selection
    #          /plot_learning_curve.html
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid(b=True, which="major", linestyle="-")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, "o-",
             color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, "o-",
             color="g", label="Cross-validation score")
    plt.legend(loc="best")

    return None


def plot_decision_surfaces(dataset, y, indices, cmap='rainbow'):
    # Inspired by Python Data Science Handbook pg 423
    n_classes = len(np.unique(y))

    # Load data
    for pairidx, pair in enumerate([[indices[0], indices[1]],
                                    [indices[0], indices[2]],
                                    [indices[0], indices[3]],
                                    [indices[1], indices[2]],
                                    [indices[1], indices[3]],
                                    [indices[2], indices[3]]]):
        # We only take the two corresponding features
        x = dataset.iloc[:, pair]

        # Train
        model.fit(x, y)

        # Plot the decision boundary
        plt.subplot(2, 3, pairidx + 1)

        # Plot the training points
        scat = plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, cmap=cmap,
                           clim=(y.min(), y.max()), edgecolor="black", s=15)

        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        xx, yy = np.meshgrid(np.linspace(*xlim, 200),
                             np.linspace(*ylim, 200))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # Create a color plot with the results
        plt.contourf(xx, yy, Z, alpha=0.3,
                     levels=np.arange(n_classes + 1) - 0.5,
                     cmap=cmap, zorder=1)

        plt.xlabel(dataset.columns.values[pair[0]])
        plt.ylabel(dataset.columns.values[pair[1]])
        plt.legend(*scat.legend_elements(), loc="lower right", borderpad=0,
                   handletextpad=0, title="Classes")
        plt.grid()

        plt.suptitle("Decision surface of a decision tree "
                     "using paired features")

        plt.axis("tight")

    return None


# Selection of dataset
print("---------------------------------")
print("Select your dataset:")
print("---------------------------------")
print("1 Simone-generated dataset Promossi/Bocciati")
print("2 Alessandro cmc dataset")
print("3 Alessandro cmc with binary output (use/no use)")

data_inp = input("Selected number: ")
if data_inp == "1":
    dataset = pd.read_csv(join(dirname(sys.argv[0]), "promossiML.csv"))
elif data_inp == "2":
    dataset = pd.read_csv(join(dirname(sys.argv[0]), "cmc.arff"),
                          header=None,
                          skiprows=list(range(0, 14)),
                          names=["wife age", "wife education",
                                 "husband education", "children born",
                                 "wife religion", "wife working?",
                                 "husband occupation", "standard of living",
                                 "media exposure", "contraceptive method"])
    dataset["contraceptive method"] -= 1
else:
    dataset = pd.read_csv(join(dirname(sys.argv[0]), "cmc.arff"),
                          header=None,
                          skiprows=list(range(0, 14)),
                          names=["wife age", "wife education",
                                 "husband education", "children born",
                                 "wife religion", "wife working?",
                                 "husband occupation", "standard of living",
                                 "media exposure", "contraceptive method"])
    dataset["contraceptive method"] = (dataset["contraceptive method"]
                                       .apply(lambda x: np.where(x == 3,
                                                                 2, x)))
    dataset["contraceptive method"] -= 1

# Shuffling of dataset
shuffle = False
print("---------------------------------")
if input("Do you want the shuffling of data? [y/n]: ") != "n":
    dataset = dataset.sample(frac=1)
    dataset = dataset.reset_index(drop=True)
    shuffle = True

# First n-1 columns are attributes x, last column is the target prediction y
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting of data into Train and Test datasets
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=2/10,
                                                shuffle=shuffle,
                                                random_state=0)
mean_on_train = xTrain.mean(axis=0)
std_on_train = xTrain.std(axis=0)
xTrain = (xTrain - mean_on_train)/std_on_train
xTest = (xTest - mean_on_train)/std_on_train

# Selection of model
print("---------------------------------")
print("Select your favourite classifier:")
print("---------------------------------")
print("1 MultiLayer Perceptron Classifier")
print("2 Decision Tree Classifier")
print("3 Support Vector Machines")
print("4 Logistic Regression")
print("5 Gaussian Naive Bayes")
print("6 Forrest Classifier")

mod_inp = input("Selected number: ")
if mod_inp == "1":
#    model = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(25,))
    model = MLPClassifier(solver='adam', activation='tanh', max_iter=2000,
                          hidden_layer_sizes=(100, 100), alpha=1e-4)
elif mod_inp == "2":
    model = DecisionTreeClassifier()
elif mod_inp == "3":
    model = svm.SVC(gamma="auto")
elif mod_inp == "4":
    model = LogisticRegression(random_state=0, solver="lbfgs",
                               multi_class="multinomial")
elif mod_inp == "5":
    model = GaussianNB()
else:
    model = ExtraTreesClassifier(max_depth=6, n_estimators=250)

# We introduce the Forest Classifier model to compute the attribute ranking
forest = ExtraTreesClassifier(n_estimators=250)

# Model fit on the training dataset
model.fit(xTrain, yTrain)
forest.fit(xTrain, yTrain)

# Prediction of the selected model on the training and test datasets
predictTrain = model.predict(xTrain)
predictTest = model.predict(xTest)

# Cross Validation (5 splits = 20% test/80% train)
nsplits = 5
cross = cross_val_score(model, x, y, cv=nsplits)

# Printing of the accuracies and confusion matrices
print("---------------------------------")
print("Accuracy and confusion matrix")
print("---------------------------------")
print("TRAIN accuracy:\t%.3f" % (accuracy_score(yTrain, predictTrain)))
print(confusion_matrix(yTrain, predictTrain))
print("---------------------------------")
print("TEST accuracy:\t%.3f" % (accuracy_score(yTest, predictTest)))
print(confusion_matrix(yTest, predictTest))
print("---------------------------------")
print("CV accuracy:\t%.3f +/- %.3f" % (np.mean(cross), np.std(cross)))

# Evaluation of the attribute ranking
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
indices_label = dataset.columns[indices]

# Printing and plot of the attribute ranking
print("---------------------------------")
print("Attribute ranking:")
print("---------------------------------")

for f in range(xTrain.shape[1]):
    print("%d. attribute %d (%f) = %s" % (f + 1, indices[f],
                                          importances[indices[f]],
                                          dataset.columns.values[indices[f]]))

plt.figure(figsize=(7, 5))
plt.title("Attribute ranking")
plt.bar(range(xTrain.shape[1]), importances[indices], color="r",
        yerr=std[indices], align="center")
plt.xticks(range(xTrain.shape[1]), indices_label, rotation=30)
plt.xlim([-1, xTrain.shape[1]])
plt.grid()
plt.tight_layout()
plt.savefig("AttributeRanking.png")

plt.figure()
plot_learning_curve(model, "Learning Curves", x, y, cv=nsplits)
plt.savefig("LearningCurve.png")

plt.figure()
plot_confusion_matrix(yTest, predictTest, classes=np.sort(np.unique(y)))
plt.savefig("ConfusionMatrixTestSet.png")

plt.figure(figsize=(12, 7))
plot_decision_surfaces(dataset, y, indices)
plt.savefig("DecisionSurfaces.png")
plt.show()
