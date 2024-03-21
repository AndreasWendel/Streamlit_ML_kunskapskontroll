import numpy as np
from sklearn.datasets import fetch_openml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle
import joblib
import streamlit as st
import tkinter
from tkinter import filedialog
# global parameter

def kmeans_predict_score (model,reference_labels,X_data,y_data):
    """
    konverterar klustrar till deras reference och sedan jämnför med target (y_data)
    :param model: tränad modell
    :param y_train: datan modellen är lärd ifrån
    :param x_data: data att predicta
    :param y_data: target till x_data
    :return: accuracy
    """
    predicted_cluster = model.predict(X_data)
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
    return (check.sum()/len(check))

def kmeans_predictions(model,reference_labels,X_data,y_data):
    predicted_cluster = model.predict(X_data)
    labeled_number = []
    for i in range(len(predicted_cluster)):
        labeled_number.append(reference_labels[predicted_cluster[i]])

    return labeled_number




def retrieve_info(model,y_train):
    #nyligen switchat parametrar kolla inputs om fel
    reference_labels = {}
    # For loop ,går igenom varje label till varje cluster label
    for i in np.unique(model.labels_):
        index = np.where(model.labels_ == i,1,0)
        num = np.bincount(y_train[index==1]).argmax()
        reference_labels[i] = num
    return reference_labels


def score (model,y_train):
    """
    ger en accuracy på hur mycket "rätt" datan sätter i ett kluster.
    :param model: kmeans model
    :param y_train: target till träningds data
    :return: score av gissad kluster till rätt target
    """
    number_labels = np.random.rand(len(model.labels_))
    reference_labels = retrieve_info(model,y_train)
    for i in range(len(model.labels_)):
        number_labels[i] = reference_labels[model.labels_[i]]
    return accuracy_score(number_labels,y_train)


def calculate_metrics(model,output):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html
    """
    print('‘Number of clusters is {}’'.format(model.n_clusters))
    print('‘Inertia : {}’'.format(model.inertia_))
    print('‘Homogeneity :       {}’'.format(metrics.homogeneity_score(output,model.labels_)))

def calculate_metrics_list(model, output):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html
    """
    metric = [model.n_clusters,model.inertia_,metrics.homogeneity_score(output, model.labels_),score(model, y_train)]

    return metric

def display_confusion_matrix(y_test, y_pred):
    """
    :param y_test: true data
    :param y_pred: pred date
    :return: plot
    """
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()



state = 42

mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

X = mnist["data"]
y = mnist["target"].astype(np.uint8)

X_myscale = X/255

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X.min(),X.max())
print(X_myscale.min(),X_myscale.max())
print(X_scaled.min(),X_scaled.max())#kolla matte bakom detta, kan vara ej optimalt för detta problem

"""
fördela datan
randomisera datan
Testa vilken skallning som är bäst, kolla bild
max på pixel kan vara 255, kolla rbg för varför.
därför kan vi skala ner den till en data som inne håller värden mellan [0,1]
kolla X.min() och X.max()
"""

plt.imshow(X[1].reshape(28, 28), cmap=mpl.cm.binary)
plt.imshow(X_myscale[1].reshape(28, 28), cmap=mpl.cm.binary)
plt.imshow(X_scaled[1].reshape(28, 28), cmap=mpl.cm.binary)
# se även skillnad här



# testar att dem olika typer av skalad data
X_train_temp = X[:10000]
X_val_temp = X[10000:11000]

X_train_temp_scaled = X_scaled[:10000]
X_val_temp_scaled = X_scaled[10000:11000]

X_train_temp_simple_scaled = X_myscale[:10000]
X_val_temp_simple_scaled = X_myscale[10000:11000]

y_train_temp = y[:10000]
y_val_temp = y[10000:11000]


# testar alla på samma simple modell
LR = LogisticRegression(random_state=state, max_iter=500)
# vanligaste modellen att använda enligt kaggle 2017
# https://blog.exploratory.io/exploratory-weekly-update-12-3-d4b1d0f620b9
# https://www.kaggle.com/kaggle-survey-2022 2022 rapporten men dem pratar inte så mkt om modeller.
# ^orelevant till arbetet men kan va kul att se

# fit "dumpar" den gamla träningen så man kkan bara calla på fit igen
# utan att instantiera LR modellen
LR.fit(X_train_temp, y_train_temp)
joblib.dump(LR, ".\ML kunskapskontroll\Data scale diff results\LR_normal_data.pkl")
LR_normal_data = joblib.load(".\ML kunskapskontroll\Data scale diff results\LR_normal_data.pkl")
print(LR_normal_data.score(X_val_temp, y_val_temp))
y_pred_Normal_data_score = LR_normal_data.predict(X_val_temp)

# 0.887 score
LR.fit(X_train_temp_scaled, y_train_temp)
joblib.dump(LR, ".\ML kunskapskontroll\Data scale diff results\LR_scaled_data.pkl")
LR_scaled_data = joblib.load(".\ML kunskapskontroll\Data scale diff results\LR_scaled_data.pkl")
print(LR_scaled_data.score(X_val_temp_scaled, y_val_temp))
y_pred_Scaled_data = LR_scaled_data.predict(X_val_temp_scaled)
# 0.907 score oxå

LR.fit(X_train_temp_simple_scaled, y_train_temp)
joblib.dump(LR, ".\ML kunskapskontroll\Data scale diff results\LR_simple_scale_data.pkl")
LR_simple_scale_data = joblib.load(".\ML kunskapskontroll\Data scale diff results\LR_simple_scale_data.pkl")
print(LR_simple_scale_data.score(X_val_temp_simple_scaled, y_val_temp))
y_pred_Simple_scale_data = LR_simple_scale_data.predict(X_val_temp_simple_scaled)

#0.932

f1_score(y_val_temp,LR.predict(X_val_temp_scaled), average="makro")
f1_score(y_val_temp,LR.predict(X_val_temp_simple_scaled), average="marko")

display_confusion_matrix(y_val_temp, y_pred_Normal_data_score, "Normal data")
display_confusion_matrix(y_val_temp, y_pred_Simple_scale_data, "Simple scaled data")
display_confusion_matrix(y_val_temp, y_pred_Scaled_data, "Standard Scaled data")


"""
kan argumentera att datan blir enkalre för modellen att hantera
och att den dessutom inte förändrar dens bild
"""
# skala data efter bäst resultat
# obs predict att slå är 93%
"""del [X_train_temp,
    X_val_temp,
    X_train_temp_scaled,
    X_val_temp_scaled,
    X_train_temp_simple_scaled,
    X_val_temp_simple_scaled,
    y_train_temp,
    y_val_temp,
    ]

"""
"""#tränings data
X_train_scaled = X_myscale[:50000]
y_train = y[:50000]
#validering
X_val_scaled = X_myscale[50000:60000]
y_val = y[50000:60000]
#träning+validering: träna om vald modell
X_trainval = np.concatenate((X_train_scaled,X_val_scaled))
y_trainval = np.concatenate((y_train,y_val))
#test data
X_test_scaled = X_myscale[60000:70000]
y_test = y[60000:70000]
"""
#minskat då en modell är 1gig
X_train_scaled = X_myscale[:20000]
y_train = y[:20000]
#validering
X_val_scaled = X_myscale[20000:30000]
y_val = y[20000:30000]
#träning+validering: träna om vald modell
"""X_trainval = X_myscale[:55000]
y_trainval = y[20000:30000]
#test data
X_test_scaled = X_myscale[55000:70000]
y_test = y[55000:70000]"""


"""
välj modeller
3 supervised,
SVM
extra tress
1 unsupervised
k-means

vi letar efter dom bästa hyperparametrarna när vi valt modell

"""


#extra trees fitted
"""
xt_param_grid ={
    "n_estimators" : range(100, 2100, 500),
}
grid_xt.best_params_
{'n_estimators': 1310}
0.96794 men knappt någon skillnad från.
så hög n tar prestanda och tid
tog 3 år att söka igenom
acc slutade öka vid ca 710
"""
"""
xt_param_grid ={
    "n_estimators" : range(100, 2100, 500),
}
"""
"""
#detta tar väldigt lång tid
xt_param_grid ={
    "n_estimators" : (1230,1260,1300),
}
#best resultat va 1260 vi kör med det
"""
#träna val modeller. spara ladda in
extra_trees_clf = ExtraTreesClassifier(random_state=state)
#testa med 100,600,100

xt_param_grid ={
    "n_estimators" : range(100,601,100),
    "max_depth":(None, 5)
}#best param 400

#högre n ökar antal träd som ökar process
#tar inte lika lång tid
#start 14:30 prick ish 10 min
grid_xt = GridSearchCV(extra_trees_clf, xt_param_grid, cv=3, verbose=10,n_jobs = 3)
grid_xt.fit(X_train_scaled, y_train)
#grid_xt.score(X_val_scaled,y_val)
joblib.dump(grid_xt, ".\ML kunskapskontroll\grid_xt_val.pkl")

grid_xt = joblib.load(".\ML kunskapskontroll\grid_xt_val.pkl")
grid_xt.score(X_val_scaled,y_val)
#0.9604
#SVM linear classifier



svm_clf = LinearSVC(random_state=state,dual="auto")
#Select the algorithm to either solve the dual or primal optimization problem. Prefer dual=False when n_samples > n_features.
svm_param_grid ={
    "C" : [0.5, 1, 1.5, 2]
}#för liten c ökar chansen för overfitting
grid_svm = GridSearchCV(svm_clf, svm_param_grid, cv=3, verbose=10,n_jobs = 3)
grid_svm.fit(X_train_scaled, y_train) #40 min tränings process för 0.89 score :D
joblib.dump(grid_svm, ".\ML kunskapskontroll\grid_svm_val.pkl")

grid_svm = joblib.load(".\ML kunskapskontroll\grid_svm_val.pkl")
grid_svm.score(X_val_scaled,y_val)
#0.9049
#svm_clf.fit(X_train_scaled,y_train)

#vi har redan våran LR model så vi kan slänga in den i validering processen

#LR.fit(X_train_scaled,y_train)


lr_param_grid ={
    "solver": ("newton-cholesky","lbfgs","sag"),
}
grid_lr = GridSearchCV(LR, lr_param_grid, cv=3, verbose=10,n_jobs = 2)
grid_lr.fit(X_train_scaled, y_train)
joblib.dump(grid_lr, ".\ML kunskapskontroll\grid_lr_val.pkl")

grid_lr = joblib.load(".\ML kunskapskontroll\grid_lr_val.pkl")
grid_lr.score(X_val_scaled,y_val)
#0.9132


#{'solver': 'lbfgs'} minimal skkillnad på solvers
#eftersom vi hanterar en klassivifering så behandlar vi problemet som en binärklassificering
#ovr jämnför om det är den siffran eller inte, ref power point
print(grid_xt.score(X_val_scaled,y_val))
print(grid_svm.score(X_val_scaled,y_val))
print(grid_lr.score(X_val_scaled,y_val))


#k-mean minibatch, dataset är för stort för kmeans.
#man kan skala ner till 10000 entries men kan bli bias då alla andra får större set att lära sig på
#ju mindre vi får kontrollera ju mer kan vi göra ett bättre uteslutande
#clusters = len(np.unique(y_train))
#vi vet att det är 10 men om datan skulle updateras up till fler så kan man behålla koden
"""mbkmeans = MiniBatchKMeans(n_clusters=clusters,random_state=state)
mbkmeans.fit(X_train_scaled)"""
# associera kluster till en label, testade närmaste centroid. dålig metod
# associerar störst mängd till kluster

"""reference_labels = {}
for i in range(len(np.unique(mbkmeans.labels_))):
    index = np.where(mbkmeans.labels_ == i,1,0)
    num = np.bincount(y_train[index==1]).argmax()
    reference_labels[i] = num
print(reference_labels)
#kluster 0 innehåller 2 osv. man ser snabbt ett problem
#grid searcha med flera kluster, teori att fler kluster kan hjälpa datorn se vad för typ
number_labels = np.random.rand(len(mbkmeans.labels_))

for i in range(len(mbkmeans.labels_)):
    number_labels[i] = reference_labels[mbkmeans.labels_[i]]

number_labels[:10]
print(accuracy_score(number_labels,y_train))"""


#score(mbkmeans,y_train)
#0.53062
#vi testar att optimera inertia, kan gå om man ökar antal kluster men kan bli fel eller överfittad


gridmbkmeans = MiniBatchKMeans(random_state=state)

param_grid ={
    "n_clusters" : range(100, 1601, 300),
}

grid_mbk = GridSearchCV(gridmbkmeans, param_grid, cv=3, verbose=10,n_jobs = 2)
grid_mbk.fit(X_train_scaled)

joblib.dump(grid_mbk, ".\ML kunskapskontroll\grid_mbk_val.pkl")

grid_mbk = joblib.load(".\ML kunskapskontroll\grid_mbk_val.pkl")
#trots att gridsearch enkelt söker igenom bra hyperparametrar så kan vi inte på något bra set setta en label på klustrar
#om du ska predicta en data point använd .reshape(1,-1)
#ex X_test_scaled[0].reshape(1,-1)
#extra_trees_clf.predict(X_test_scaled[0].reshape(1,-1))

mbkgridresults = pd.DataFrame(grid_mbk.cv_results_)
 #högst cluster kommer allit ha bättre inertia med när faller den av
temp1 = pd.DataFrame()
temp1["inertia"] = (mbkgridresults["mean_test_score"]*-1)
temp1["n_clusters"] = mbkgridresults['param_n_clusters'].astype(int)
temp1['deriv'] = (temp1['inertia'] - temp1['inertia'].shift(1)) / (temp1['n_clusters'] - temp1['n_clusters'].shift(1))
temp1['2deriv'] = (temp1['n_clusters'] - temp1['n_clusters'].shift(1)) / (temp1['deriv'] - temp1['deriv'].shift(1))
temp1['inertia'].plot()

fig_inertia, ax = plt.subplots()
ax.plot(temp1['n_clusters'], temp1['inertia'], marker='o', linestyle='-')
ax.set_xlabel('n_clusters')
ax.set_ylabel('inertia')
ax.set_title('inertia decrease by cluster')

#vi ser att förändring hastigheten börjar stabilisera sig mellan 700-1300 kluster
#vi förväntar oss att inertia minskar med antal kluster men vill också ha så lite kluster som möjligt.
#vi väljer en bra punkt där derivatan börjar stabilisera sig,
#om modellen medför bra resultat kan vi optimer den senare

cluster_number = [700,800,900,1100,1300,1400,1600]
inertia_list = []

for i in cluster_number:
    # Initialize the K-Means model
    kmeanstest = MiniBatchKMeans(n_clusters=i,random_state=state)
    # Fitting the model to training set
    kmeanstest.fit(X_train_scaled)
    # Calculating the metrics
    inertia_list.append(kmeanstest.inertia_)
    calculate_metrics(kmeanstest, y_train)
    # Calculating reference_labels
    print('Accuracyscore: {}'.format(score(kmeanstest,y_train)))


inertiaDF = pd.DataFrame(data = inertia_list,index = cluster_number)
inertiaDF.plot()




for i in cluster_number:
    # Initialize the K-Means model
    kmeanstest = MiniBatchKMeans(n_clusters=i, random_state=state)
    # Fitting the model to training set
    kmeanstest.fit(X_train_scaled)
    joblib.dump(kmeanstest, ".\ML kunskapskontroll\kmeans cluster test\kmeanstest"+str(i)+".pkl")


temp = []
for i in cluster_number:
    kmeanstest = joblib.load(".\ML kunskapskontroll\kmeans cluster test\kmeanstest"+str(i)+".pkl")
    temp.append(calculate_metrics_list(kmeanstest, y_train))

metric = pd.DataFrame(temp,columns=["Cluster","inertia","Homogeneity","Accuracy score"])
metric.head()




"""
big problemo, den nya skalade datan i kmeans. gör att vissa kluster inte är tilldelade data och då inte get en reference till datan.
när man väl vill prediktera datan så sätter modellen den nya X_val datan i ett kluster som inte har fått någon reference vilket leder till att den inte vet vad de siffran är för något.
lösning? andra skalare, mer data, minska antal kluster.
"""


label_comparison = pd.DataFrame()
for i in cluster_number:
    kmeanstest = MiniBatchKMeans(n_clusters=i, random_state=state)
    kmeanstest.fit(X_train_scaled)
    print("n_cluster:",i)
    print("labeled clusters:",len(np.unique(kmeanstest.labels_)))


for i in cluster_number:
    kmeanstest = MiniBatchKMeans(n_clusters=i, random_state=state)
    kmeanstest.fit(X_train_scaled)
    df = {"n_clusters":i, "labeled clusters_scaler":len(np.unique(kmeanstest.labels_))}
    print("n_cluster:",i)
    print("labeled clusters:",len(np.unique(kmeanstest.labels_)))
    label_comparison = label_comparison._append(df, ignore_index=True)

label_comparison.to_csv("ML kunskapskontroll\cluster_label_comparison.csv")

#vi kan ser den linjärt minskar, vilket kan betyda att vi inte har funnit en armbåge, eller att vi redan är förbi armbågen
#om man hänvisar till gridsearch resultatet kan vi anta att vi är förbi armbågen.
#samt att ökning av kluster kommer öka prestanda tid och arbete något enormt.
#vi kan oxå se att ökningen avv accuracy minskar signifikt och att vi ändå inte når upp till extra trees standard.
mbkmeans = MiniBatchKMeans(n_clusters=700,random_state=state)
mbkmeans.fit(X_train_scaled)

joblib.dump(mbkmeans, ".\ML kunskapskontroll\opti_mbkmeans.pkl")


grid_lr = joblib.load(".\ML kunskapskontroll\grid_lr_val.pkl")
grid_svm = joblib.load(".\ML kunskapskontroll\grid_svm_val.pkl")
grid_xt = joblib.load(".\ML kunskapskontroll\grid_xt_val.pkl")
opti_mbkm = joblib.load(".\ML kunskapskontroll\opti_mbkmeans.pkl")
reference_labels = retrieve_info(opti_mbkm,y_train)




print('Kmeans Score: {}'.format(kmeans_predict_score(opti_mbkm,reference_labels,X_val_scaled,y_val)))
print('Extra Trees Score: {}'.format(grid_xt.score(X_val_scaled,y_val)))
print('SVM Score: {}'.format(grid_svm.score(X_val_scaled,y_val)))
print('Logistic Regression Score: {}'.format(grid_lr.score(X_val_scaled,y_val)))

hold = kmeans_predictions(opti_mbkm,reference_labels,X_val_scaled,y_val)

f1_score(y_val,grid_xt.predict(X_val_scaled), average="micro")
f1_score(y_val,hold, average="micro")

display_confusion_matrix(y_val,hold)
display_confusion_matrix(y_val,grid_xt.predict(X_val_scaled))
display_confusion_matrix(y_val,grid_svm.predict(X_val_scaled))
display_confusion_matrix(y_val,grid_lr.predict(X_val_scaled))



"""evaluerings processen"""
"""
    obs läs
    om pkl filen är för stor, så kan du träna modellen själv. det tar inte lika lång tid som man tror
    kör bara hela koden ner till joblib
"""
state = 42

mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

X = mnist["data"]
y = mnist["target"].astype(np.uint8)
X_myscale = X/255
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_myscale, y, test_size=0.20, random_state=state)

ev_extra_trees = ExtraTreesClassifier(n_estimators=200 ,max_depth=None ,random_state = state)
ev_extra_trees.fit(X_train_full,y_train_full)
joblib.dump(ev_extra_trees, ".\ML kunskapskontroll\Final_EX_tree.pkl",compress=3)
#kör hela vägen ner hit om du vill träna modellen själv

final_EX_tree = joblib.load(".\ML kunskapskontroll\Final_EX_tree.pkl")
print('Final model score: {}'.format(final_EX_tree.score(X_test_full,y_test_full)))
display_confusion_matrix(y_test_full,final_EX_tree.predict(X_test_full))



"""ev_extra_trees.fit(shuffled_X,shuffled_y)
ev_extra_trees.score(X_test_scaled,y_test)
"""
# streamlit run '.\ML kunskapskontroll\MNIST Kunskapskontroll 2 AW.py'
