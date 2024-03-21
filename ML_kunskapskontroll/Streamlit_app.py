import numpy as np
from sklearn.datasets import fetch_openml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
import cv2
import streamlit as st


# global parameter

def kmeans_predict_score(model, reference_labels, X_data, y_data):
    """
    konverterar klustrar till deras reference och sedan jämnför med target (y_data)
    :param X_data: data att predicta
    :param reference_labels: reference labels
    :param model: tränad modell
    :param y_data: target till x_data
    :return: accuracy
    """
    predicted_cluster = model.predict(X_data)  # .reshape(1,-1)
    labeled_number = []
    for i in range(len(predicted_cluster)):
        labeled_number.append(reference_labels[predicted_cluster[i]])

    check = []
    for i in range(len(labeled_number)):
        if labeled_number[i] == y_data[i]:
            check.append(True)
        else:
            check.append(False)

    check = np.array(check)
    return (check.sum() / len(check))


def kmeans_predictions(model, reference_labels, X_data, y_data):
    predicted_cluster = model.predict(X_data)
    labeled_number = []
    for i in range(len(predicted_cluster)):
        labeled_number.append(reference_labels[predicted_cluster[i]])

    return labeled_number


def retrieve_info(model, y_train):
    # nyligen switchat parametrar kolla inputs om fel
    reference_labels = {}
    # For loop ,går igenom varje label till varje cluster label
    for i in np.unique(model.labels_):
        index = np.where(model.labels_ == i, 1, 0)
        num = np.bincount(y_train[index == 1]).argmax()
        reference_labels[i] = num
    return reference_labels


def score(model, y_train):
    """
    ger en accuracy på hur mycket "rätt" datan sätter i ett kluster.
    :param model: kmeans model
    :param y_train: target till träningds data
    :return: score av gissad kluster till rätt target
    """
    number_labels = np.random.rand(len(model.labels_))
    reference_labels = retrieve_info(model, y_train)
    for i in range(len(model.labels_)):
        number_labels[i] = reference_labels[model.labels_[i]]
    return accuracy_score(number_labels, y_train)


def calculate_metrics_list(model, output):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html
    """
    metric = [model.n_clusters, model.inertia_, metrics.homogeneity_score(output, model.labels_), score(model, y_train)]

    return metric


def display_confusion_matrix(y_test, y_pred):
    """
    :param y_test: true data
    :param y_pred: pred date
    :return: plot
    """
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()




# plt.imshow(normalized_image.reshape(28, 28), cmap=mpl.cm.binary)


def st_get_image(uploaded_image):
    image_bytes = uploaded_image.read()

    # Decode the image from bytes using OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    image = 255 - image
    resized_image = cv2.resize(image, (28, 28))  # Resize image to 28x28
    normalized_image = resized_image / 255.0  # Normalize pixel
    flattened_image = normalized_image.flatten()
    flattened_image = np.reshape(flattened_image, (1, 784))  # Reshape
    return flattened_image


def fig_imshow(data):
    fig, ax = plt.subplots()
    ax.imshow(data[0].reshape(28, 28), cmap=mpl.cm.binary)
    return fig


def fig_display_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ax.set_title(title)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    return fig


def predict_image(transformed_image, model):
    prediction = model.predict(transformed_image)
    return prediction[0]


# streamlit run '.\ML kunskapskontroll\Streamlit app.py'

nav = st.sidebar.radio("Navigation Menu", ["Author", "Data & Modelling", "predict"])

if nav == "Author":
    st.title("Written by")
    st.write("Andreas Wendel")
    st.write("EC utbildning")
    st.write("[https://github.com/AndreasWendel](https://github.com/AndreasWendel)")

if nav == "Data & Modelling":

    state = 42

    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

    X = mnist["data"]
    y = mnist["target"].astype(np.uint8)

    X_myscale = X / 255

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(X.min(), X.max())
    print(X_myscale.min(), X_myscale.max())
    print(X_scaled.min(), X_scaled.max())

    fig_show_X, ax_show_X = plt.subplots()
    ax_show_X.imshow(X[1].reshape(28, 28), cmap=mpl.cm.binary)

    fig_show_X_myscale, ax_show_X_myscale = plt.subplots()
    plt.imshow(X_myscale[1].reshape(28, 28), cmap=mpl.cm.binary)

    fig_show_X_scaled, ax_show_X_scaled = plt.subplots()
    plt.imshow(X_scaled[1].reshape(28, 28), cmap=mpl.cm.binary)
    # se även skillnad här

    # testar att dem olika typer av skalad data
    X_val_temp = X[10000:11000]
    X_val_temp_scaled = X_scaled[10000:11000]
    X_val_temp_simple_scaled = X_myscale[10000:11000]
    y_val_temp = y[10000:11000]

    # testar alla på samma simple modell
    # LR = LogisticRegression(random_state=state, max_iter=500)
    # vanligaste modellen att använda enligt kaggle 2017
    # https://blog.exploratory.io/exploratory-weekly-update-12-3-d4b1d0f620b9
    # https://www.kaggle.com/kaggle-survey-2022 2022 rapporten men dem pratar inte så mkt om modeller.
    # ^orelevant till arbetet men kan va kul att se

    LR_normal_data = joblib.load("./ML_kunskapskontroll/Data scale diff results/LR_normal_data.pkl")
    Normal_data_score = LR_normal_data.score(X_val_temp, y_val_temp)
    y_pred_Normal_data_score = LR_normal_data.predict(X_val_temp)

    LR_simple_scale_data = joblib.load("./ML_kunskapskontroll/Data scale diff results/LR_simple_scale_data.pkl")
    Simple_scale_data = LR_simple_scale_data.score(X_val_temp_simple_scaled, y_val_temp)
    y_pred_Simple_scale_data = LR_simple_scale_data.predict(X_val_temp_simple_scaled)

    LR_scaled_data = joblib.load("./ML_kunskapskontroll/Data scale diff results/LR_scaled_data.pkl")
    Scaled_data = LR_scaled_data.score(X_val_temp_scaled, y_val_temp)
    y_pred_Scaled_data = LR_scaled_data.predict(X_val_temp_scaled)

    f1_score_normal_data = f1_score(y_val_temp, y_pred_Normal_data_score, average="macro")
    f1_score_Simple_scaled_data = f1_score(y_val_temp, y_pred_Simple_scale_data, average="macro")
    f1_score_Scaled_data = f1_score(y_val_temp, y_pred_Scaled_data, average="macro")

    # print(f1_score_normal_data)
    # print(f1_score_Simple_scaled_data)
    # print(f1_score_Scaled_data)

    fig_cm_normal = fig_display_confusion_matrix(y_val_temp, y_pred_Normal_data_score, "Normal data")
    fig_cm_simple_scale = fig_display_confusion_matrix(y_val_temp, y_pred_Simple_scale_data, "Simple scaled data")
    fig_cm_scaled = fig_display_confusion_matrix(y_val_temp, y_pred_Scaled_data, "Standard Scaled data")
    # skala data efter bäst resultat
    # obs predict att slå är 93%
    del [
        X_val_temp,
        X_val_temp_scaled,
        X_val_temp_simple_scaled,
        y_val_temp
    ]

    # minskat då en modell är 1gig
    X_train_scaled = X_myscale[:20000]
    y_train = y[:20000]
    # validering
    X_val_scaled = X_myscale[20000:30000]
    y_val = y[20000:30000]
    # träning+validering: träna om vald modell

    # val modeller

    grid_mbk = joblib.load("./ML_kunskapskontroll/grid_mbk_val.pkl")

    mbkgridresults = pd.DataFrame(grid_mbk.cv_results_)
    # högst cluster kommer allit ha bättre inertia med när faller den av
    temp1 = pd.DataFrame()
    temp1["inertia"] = (mbkgridresults["mean_test_score"] * -1)
    temp1["n_clusters"] = mbkgridresults['param_n_clusters'].astype(int)
    temp1['deriv'] = (temp1['inertia'] - temp1['inertia'].shift(1)) / (
            temp1['n_clusters'] - temp1['n_clusters'].shift(1))
    temp1['2deriv'] = (temp1['n_clusters'] - temp1['n_clusters'].shift(1)) / (temp1['deriv'] - temp1['deriv'].shift(1))
    temp1['inertia'].plot()

    fig_inertia, ax_I = plt.subplots()
    ax_I.plot(temp1['n_clusters'], temp1['inertia'], marker='o', linestyle='-')
    ax_I.set_xlabel('n_clusters')
    ax_I.set_ylabel('inertia')
    ax_I.set_title('inertia decrease by cluster')

    cluster_number = [700, 800, 900, 1100, 1300, 1400, 1600]
    inertia_list = []
    score_list = []
    temp = []
    for i in cluster_number:
        kmeanstest = joblib.load("./ML_kunskapskontroll/kmeans cluster test/kmeanstest" + str(i) + ".pkl")
        temp.append(calculate_metrics_list(kmeanstest, y_train))

    metric = pd.DataFrame(temp, columns=["Cluster", "inertia", "Homogeneity", "Accuracy score"])
    metric.head()

    coef = np.polyfit(metric['Cluster'], metric['inertia'], 1)
    poly1d_fn = np.poly1d(coef)
    fig_inertia_optimised, ax_IO = plt.subplots()
    ax_IO.plot(metric['Cluster'], metric['inertia'], marker='o', linestyle='-')
    ax_IO.plot(metric['Cluster'], poly1d_fn(metric['Cluster']), '-k')
    ax_IO.set_xlabel('Cluster')
    ax_IO.set_ylabel('inertia')
    ax_IO.set_title('inertia decrease by cluster')

    label_comparison = pd.read_csv("./ML_kunskapskontroll/cluster_label_comparison.csv")
    label_comparison.set_index("n_clusters", inplace=True)
    label_comparison.drop("Unnamed: 0", inplace=True, axis=1)

    grid_lr = joblib.load("./ML_kunskapskontroll/grid_lr_val.pkl")
    grid_svm = joblib.load("./ML_kunskapskontroll/grid_svm_val.pkl")
    grid_xt = joblib.load("./ML_kunskapskontroll/grid_xt_val.pkl")
    opti_mbkm = joblib.load("./ML_kunskapskontroll/opti_mbkmeans.pkl")
    reference_labels = retrieve_info(opti_mbkm, y_train)
    kmeans_predict = kmeans_predictions(opti_mbkm, reference_labels, X_val_scaled, y_val)

    dict = {"model": ["Logistic Regression", "Extra Trees", "LinearSVC", "MBKmeans"],
            "accuracy score": [grid_lr.score(X_val_scaled, y_val),
                               grid_xt.score(X_val_scaled, y_val),
                               grid_svm.score(X_val_scaled, y_val),
                               kmeans_predict_score(opti_mbkm, reference_labels, X_val_scaled, y_val)],
            "F1 score": [f1_score(y_val, grid_lr.predict(X_val_scaled), average="macro"),
                         f1_score(y_val, grid_xt.predict(X_val_scaled), average="macro"),
                         f1_score(y_val, grid_svm.predict(X_val_scaled), average="macro"),
                         f1_score(y_val, kmeans_predict, average="macro")]
            }
    scoreDF = pd.DataFrame(dict)

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_myscale, y, test_size=0.20,
                                                                            random_state=42)

    final_EX_tree = joblib.load("./ML_kunskapskontroll/Final_EX_tree.pkl")  # litet f skapar string format....

    print('Final model score: {}'.format(final_EX_tree.score(X_test_full, y_test_full)))
    from tkinter.filedialog import askopenfilename

    y_test_pred = final_EX_tree.predict(X_test_full)
    # display_confusion_matrix(y_test_full, y_test_pred)

    st.title("Data & Machine Learning Modelling")

    st.header("Data manipulations")
    st.subheader("difference in scaling the data")
    st.write('Scaling data is important but lets look at the difference in scaled data')
    st.write("min:", X.min(), ",", "max:", X.max(), "values for non scaled data")
    st.write("Pixels have a standard value of 0-255 (in rbg setting), by dividing by 255 we get value between 0,1")
    st.write("min:", X_myscale.min(), ",", "max:", X_myscale.max(), "values scaled by i/255")
    st.write(
        "This time we scale the data using standardscaler which will normalize your features with a mean=0 and "
        "standard deveiation=1")
    st.write("min:", X_scaled.min(), ",", "max:", X_scaled.max(), "values scaled by standard scaler")

    st.subheader("different outcome using differently scaled data")
    st.write("Use a small sample of the data to train a modell using differently scaled data")
    st.write("Using a Logistical Regression modell for evaluating the data")
    st.write("read: https://blog.exploratory.io/exploratory-weekly-update-12-3-d4b1d0f620b9")
    st.write("There are 2 ways of evaluating the modells with different data")
    st.write("we can use a simple n_right_predictions/all_predictions OR we can use f1 score")
    st.write("f1 score uses a confusion matrix to calculate recall and precision")
    st.write(
        "read: https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult")

    st.subheader("using normal score")
    st.write("score for non scaled data:", Normal_data_score)
    st.write("score for data scaled by i/255:", Simple_scale_data)
    st.write("score for data scaled by standard scaler:", Scaled_data)

    st.subheader("using F1 score, param macro")
    st.write("score for non scaled data:", f1_score_normal_data)
    st.write("score for data scaled by i/255:", f1_score_Simple_scaled_data)
    st.write("score for data scaled by standard scaler:", f1_score_Scaled_data)

    st.subheader("we can a detail look into the predictions using a confusion matrix")
    st.write("non scaled data")
    st.pyplot(fig_cm_normal)
    st.write("data scaled by i/255")
    st.pyplot(fig_cm_simple_scale)
    st.write("data scaled by standard scaler")
    st.pyplot(fig_cm_scaled)

    st.write("There is a slight difference in the matrix for each of the data but nothing major")
    st.subheader("lets quickly look at the first iteration of the data as a visual")
    st.write("normal data visualised")
    st.pyplot(fig_imshow(X))
    st.write("data scaled by i/255visualised")
    st.pyplot(fig_imshow(X_myscale))
    st.write("data scaled by standard scalervisualised")
    st.pyplot(fig_imshow(X_scaled))
    st.write("Note that when using the standard scaler we get a computer image that looks quite blurred. "
             "Some might argue that this makes the data bad and other could argue that it trains the model on noise. "
             "Noise could be colors from images that are not plain white which disturbs the images")
    st.write("we make a simple decision to scale the data into [0,1] as that seem to give the best score")

    st.header("Valuating models")
    st.write(
        "we create a simple valuation process using 3 supervised learning models and 1 unsupervised learning model")
    st.write("We implement a different kinds of models and train them using different hyperparameters")
    st.write(
        "To make the process easier we use Sklearns gridsearchcv, which divides the whole data into training and "
        "validation part for us")
    st.write("as to not make the training process too time consuming we limit the amount of different parameters")
    st.write("each addition in the parameter exponentially increases the training time as iterations increase")

    st.header("Supervised learning")
    st.subheader("Logistic Regression")
    st.write(
        "Logistic Regression is on of the most used Classifier models and we have already implement it when valuating "
        "data so we might aswell continue to experiment with it"
        "Logistic Regression is a statistical method used for binary classification tasks, "
        "where the goal is to predict the probability that a given input belongs to a particular class"
        "Binary output in this case will be a is or is not kind of definition. ex.is(number 4 = 1 or 0). "
        "The model has a parameter called solver which specifies the algorithm used to solve the optimization problem "
        "during model training")
    st.write("Hyperparameters Chosen for LR", grid_lr.param_grid)
    st.write("Results")
    st.dataframe(grid_lr.cv_results_)

    st.subheader("Extra trees")
    st.write("Extra Trees, short for Extremely Randomized Trees, "
             "is an ensemble learning method used for classification and regression tasks "
             "It's similar to Random Forests but with a key difference: "
             "Extra Trees introduces additional randomness during the construction of each decision tree in the "
             "ensemble.")
    st.write("Hyperparameters Chosen for Extra trees", grid_xt.param_grid)
    st.write("Results")
    st.dataframe(grid_xt.cv_results_)

    st.subheader("LinearSVC")
    st.write("Linear Support Vector Classification (LinearSVC) is a type of Support Vector Machine (SVM) classifier. "
             "that aims to find the best hyperplane that separates the classes in the input space. "
             "completly simulare to normal SVC (Support Vector Classification) but using a linear kernel. "
             "Keep in mind that SVC is usually bad at handling big datasets so to decrease training time we can have "
             "a linear kernel model instead as the model creates hyperplanes change the parameter c to allow for "
             "margin violations which means that we allow for more flexibility")
    st.write("Hyperparameters Chosen for LinearSVC", grid_svm.param_grid)
    st.write("Results")
    st.dataframe(grid_svm.cv_results_)

    st.header("Unsupervised learning")
    st.subheader("K-means")
    st.write(
        "K-means is trained differently from the other models as it is unsupervised meaning it will not have a target "
        "label. Instead it aims to distribut the data into clusters. Choosing the number of clusters is the tricky "
        "part when handling cluster models. You could argue that we have 10 numbers and so we should have 10 "
        "clusters. But there is another argument which says that you can have more clusters as each clusters "
        "represents a way of writing a number. "
        "Aswell as k-means being handled differenly it is also evaluated by something called inertia which simply "
        "calculates the distance from a datapoint to its klusters center "
        "That means that a lower inertia gives a better ''score'' but in sklearn there is a rule which strictly says "
        "that greater is better which is why gridsearch will return negative inertia. "
        "valuating kmeans in a fair way is quite difficult and we will dwell into it further down. "
        "Note that we are using Minibatchkmeans which takes small random batches of the data instead of all of it so "
        "that we can decrease training time")
    st.write("Hyperparameters Chosen for Kmeans", grid_mbk.param_grid)
    st.write("Results")
    st.dataframe(grid_mbk.cv_results_)
    st.write("you see the correlation that when n_clusters increase inertia decreases "
             "which is quite self explanatory as we get less distance between "
             "a decent way to look at inertia is to see the decrease which cluster and find the elbow")

    st.header("Kmeans Inertia")
    st.subheader("inertia plot")
    st.pyplot(fig_inertia)
    st.write("note that the elbow start around 400 clusters but lets look at speed of change.")
    st.dataframe(temp1)
    st.write(
        "note that the coefficient between the points are slowly decreasing and id argue that around 700 we get the "
        "first point where the increase of cluster start to become meaningless")
    st.write(
        "we create a new test without gridsearchcv and train kmeans models using a list of clusters and compare the "
        "results")

    st.subheader("inertia plot")
    st.pyplot(fig_inertia_optimised)
    st.write("note that when we are at this level of cluster inertia seems to stabilizes, "
             "The coef is about to reach a linear level and we can't see the flipped exponetial decrease like before")

    st.header("dataframe from kmeans analysis")
    st.write(
        "The other problem with kmeans is that inertia doesn't really say that much to us other than that we want it "
        "low so instead we can use the target data we have to label each cluster and then compare it with the true "
        "targets.")
    st.write(
        "As per our last analysis where we train straight from the model without gridsearchcg, we are now able to "
        "call the instance and get more information such as labeled clusters, this means that we can now label "
        "clusters using different methods. As we have the target data for the trained X data we can now use that "
        "information to give each datapoint a number and then label each cluster by the most amount of numbers in "
        "that cluster. Ex lets say cluster nr1 have 4 data points = number 1 and 1 data point = number 2. that "
        "cluster is now labeled as a 1 as number 1 appears the most ")
    st.write(
        "so now that we have a reference to each label we can now see if the model put all the trained data in the "
        "right labels. Keep in mind that clusters move to new points per iteration not the data. So a 99% accuracy is "
        "not likely as some handwritten numbers could look like the wrong number ")
    st.subheader("kmeans metrics")
    st.dataframe(metric)
    st.write(
        "Homogeneity: A clustering result satisfies homogeneity if all of its clusters contain only data points which "
        "are members of a single class.")

    st.header("cluster and label problem")
    st.write(
        "a Machine learning problem which we happen to stumble uppon where some clusters doesn't have any data points "
        "labeled to them. "
        "This creates a problem where some cluster cannot be labeled as we cannot referance a data point to the "
        "cluster making it unlabeled "
        "other than the apparent issue that we have to many cluster, whenever we decide to predict new data an issue "
        "arises where some data "
        "are appointed to clusters where have failed to label them.")
    st.write(
        "For some reason even though a cluster that have literly no association, when predicting the modell may put "
        "them there anyways. "
        "This might be bc they have never ¨seen¨ such a data(written number) before")
    st.write(
        "an apparent solution might be to lower the cluster or change/increase the trained data. An increase in data "
        "would make sense but as we want to make. "
        "The valuation process as fair as possible we reside to another solution")
    st.subheader("Table of comparisons using less clusters")
    st.dataframe(label_comparison)

    st.write(
        "The best decision here is to just retrain a model that has 700 clusters as that seems to be where the model "
        "can fill each cluster and where inertia decrease seem to be slowing down. "
        "An issue might arise later that you might call overfitting where you get data the model "
        "has not seen before.")
    st.write(
        "lets leave kmeans at that, we want to make a fair valuation and putting to much time into such a problem "
        "might unfairly overtune it")

    st.header("Confusion matrix for all modells")
    st.subheader("lets look at some metrics")
    st.write("we can start by looking a ta simle confusion matrix")
    st.subheader("Logistic Regression matrix")
    st.pyplot(fig_display_confusion_matrix(y_val, grid_lr.predict(X_val_scaled),
                                           "Logistic Regression Matrix"))  # fix allt under
    st.subheader("Extra tree matrix")
    st.pyplot(fig_display_confusion_matrix(y_val, grid_xt.predict(X_val_scaled), "Extra tree matrix"))
    st.subheader("LinearSVC matrix")
    st.pyplot(fig_display_confusion_matrix(y_val, grid_svm.predict(X_val_scaled), "LinearSVC matrix"))
    st.subheader("Kmeans matrix")
    st.pyplot(fig_display_confusion_matrix(y_val, kmeans_predict, "Kmeans matrix"))

    st.subheader("Finally we can see a difference in score using normal accuracy score and F1 Score, macro")
    st.dataframe(scoreDF)

    st.header("Conclusion of validation process")
    st.write("looking first at the matrixes we can see that kmeans struggles at some numbers to predict. "
             "we can see that all models have a slight struggle when predicting 5 and 8 and all models most accuratly "
             "predict the number 1. "
             "over all Extra tress seems to be performing the best followed by both Kmeans and LR.")
    st.write("And given this information it is safe to argue that we should continue using Extra trees")
    st.dataframe(grid_xt.cv_results_)
    st.write("looking at the cv_results we can see that the top 3 parameters gives the same mean score. "
             "so to make it easier for us to store we use the one with less n_estimators as that will "
             "lower the file size and retraining time.")

    st.header("Final model")
    st.subheader(
        "Now that we have chosen a model we can retrain the model using the best parameters with the train and "
        "validation data together.")
    st.write("we can now use 80% of the full dataset and the other 20% as a evaluation set")
    st.write("finally we can get a last score of")
    st.write('Final model score: {}'.format(final_EX_tree.score(X_test_full, y_test_full)))
    st.write('Final model f1 Score: {}'.format(f1_score(y_test_full, final_EX_tree.predict(X_test_full), average="macro")))
    st.write("and a confusion matrix")
    st.pyplot(fig_display_confusion_matrix(y_test_full, final_EX_tree.predict(X_test_full), "Final Extra Trees matrix"))
    st.write("Only downside with Extra trees is that the train model is a great amount of size")

if nav == "predict":

    final_EX_tree = joblib.load("./ML_kunskapskontroll/Final_EX_tree.pkl")  # litet f skapar string format....

    st.title("Prediction Modell test")
    st.write('Try uploading an image to predict')
    st.write('image are compress to a format and dimension that the model can read. '
             'If prediction fails the compression might have lead to dataloss')
    st.write(
        "Which could lead to the model predicting wrong. Remember we have just created something digitaly that is now "
        "interacting with the real world. problems are bound to happen")
    image_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])
    if image_file is not None:
        st.write("Selected file:", image_file.name)
        st.image(image_file, caption='Uploaded Image', use_column_width=True)
        image = st_get_image(image_file)
        st.header("image")
        st.subheader("image")
        st.pyplot(fig_imshow(image))
        st.subheader("prediction")
        st.write("predicted number")
        st.write(predict_image(image, final_EX_tree))
