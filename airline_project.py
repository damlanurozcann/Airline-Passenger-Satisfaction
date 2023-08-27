#####################################################################################
###########################      IMPORTS      #######################################
#####################################################################################

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.release import classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

#!pip install catboost
#!pip install lightgbm
#!pip install xgboost

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

#####################################################################################
########################### HELPERS FUNCTIONS #######################################
#####################################################################################

# from helpers import *
# If the file is loaded, we can import it directly from here.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

"""def plot_importance(model, features, num=len(X), save=False,name="importance"):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(f'{name}.png')
"""
####################################################################################

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("datasets/train.csv")




##################################
# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
##################################

##################################
# GENEL RESİM
##################################

check_df(df)

##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols
num_cols
cat_but_car



fig = plt.figure(figsize=(8, 5))
df.satisfaction.value_counts(normalize=True).plot(kind='bar', color=['darkorange', 'steelblue'], alpha=0.9, rot=0)
plt.title('Satisfaction')
plt.show(block="True")

df["satisfaction"] = df["satisfaction"].apply(lambda x: 1 if x == "satisfied" else 0)

##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

for col in cat_cols:
    cat_summary(df, col)

for col in cat_cols:
    target_summary_with_cat(df, "satisfaction", col)

##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

for col in num_cols:
    num_summary(df, col, plot=True)


for col in num_cols:
    target_summary_with_num(df, "satisfaction", col)


def check_travel_class(df, travel_type):
    """
    Verilen veri çerçevesinde, belirli bir seyahat türüne sahip kişinin iş seyahati yapma durumunu ve seyahat sınıfını kontrol eder.

    Parameters:
        df (DataFrame): Veri çerçevesi.
        travel_type (str): Seyahat türü ("Business travel" veya "Personal Travel").

    Returns:
        str: İş seyahati durumu ve seyahat sınıfı bilgisi.
    """
    filtered_df = df[df['Type of Travel'] == travel_type]
    if filtered_df.empty:
        return "Bu seyahat türüne ait veri bulunamadı."

    business_travel = filtered_df[filtered_df['Class'] == 'Business']

    if not business_travel.empty:
        return "Bu kişi iş seyahati yapmış ve Business sınıfında uçmuş."
    else:
        return "Bu kişi iş seyahati yapmamış veya farklı bir sınıfta uçmuş."


# Örnek kullanım:
result = check_travel_class(df, "Business travel")
print(result)


def compare_satisfaction_factors(df):
    """
    Verilen veri çerçevesinde her bir seyahat sınıfı için memnuniyeti düşüren faktörleri karşılaştırır.

    Parameters:
        df (DataFrame): Veri çerçevesi.

    Returns:
        DataFrame: Memnuniyet faktörlerinin karşılaştırması.
    """
    factors = ['Inflight wifi service', 'Inflight entertainment', 'Food and drink', 'Seat comfort', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness']
    travel_classes = df['Class'].unique()

    results = {}
    for travel_class in travel_classes:
        class_df = df[df['Class'] == travel_class]
        if class_df.empty:
            results[travel_class] = "Bu seyahat sınıfına ait veri bulunamadı."
        else:
            satisfaction_factors = class_df[factors]
            satisfaction_factors_mean = satisfaction_factors.mean().sort_values(ascending=False)
            results[travel_class] = satisfaction_factors_mean

    return results

# Örnek kullanım:
result = compare_satisfaction_factors(df)
for travel_class, factors in result.items():
    print(f"Seyahat Sınıfı: {travel_class}")
    print(factors)
    print()

    sns.catplot("satisfaction", col="Customer Type", col_wrap=2, data=df, kind="count", height=2.5, aspect=1.0)
    # Grafikleri gösterin
    plt.show(block=True)

    sns.catplot(x="Flight Distance", y="Type of Travel", hue="satisfaction", col="Class", data=df, kind="bar",
                height=4.5, aspect=.8)
    plt.show(block=True)

###########
# Eksik DEĞİŞKENLERİN ANALİZİ
##################################

df.isnull().sum()
na_columns = missing_values_table(df, na_name=True)

##################################
# KORELASYON
##################################

df[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block="True")


df.corrwith(df["satisfaction"]).sort_values(ascending=False)

df.info()

####################################################################
##                      FEATURE ENGINEERING                       ##
####################################################################

##################################
# Before start, Dropping unnecessary variables, column names update, classify variables and separate target variable
##################################

df = df.drop("id", axis=1)
df = df.drop("Unnamed: 0", axis=1) # We don't need those variables.

df.columns = [col.upper() for col in df.columns]

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols = [col for col in cat_cols if col not in ["SATISFACTION"]] # Satisfaciton is our target value.

##################################
# Missing value analysis
##################################

df.isnull().sum()

missing_values_table(df)
na_cols = missing_values_table(df, na_name=True)

# missing_vs_target(df, "satisfaction", na_cols)
# The function does not work for now because "satisfaction" is not in a standardized state

df = label_encoder(df, "SATISFACTION") ###  ! (1: satisfaction) & (0: neutral or dissatisfied)

missing_vs_target(df, "SATISFACTION", na_cols)

##################################
# Filling Missing value
##################################

# Since there is a linear relationship between 'Arrival time' & 'Departure Time', and only a few missing values exist, I have chosen to fill the missing values with the corresponding 'Departure Time' values in order to preserve the linear relationship.
df["ARRIVAL DELAY IN MINUTES"].fillna(df["DEPARTURE DELAY IN MINUTES"], inplace=True)

df.isnull().sum()

##################################
# Outliers Analysis
##################################

for col in num_cols:
    print(col, check_outlier(df, col, q1=0.05, q3=0.95))

for col in  num_cols:
    print(col, check_outlier(df, col, q1=0.05, q3=0.95))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col, q1=0.05, q3=0.95))

##################################
# Feature extraction
##################################

df.loc[(df["AGE"] < 18), "NEW_AGE_CAT"] = "Child"
df.loc[(df["AGE"] >= 18) & (df["AGE"] < 35), "NEW_AGE_CAT"] = "mature"
df.loc[(df["AGE"] >= 35) & (df["AGE"] < 50), "NEW_AGE_CAT"] = "Senior"
df.loc[(df["AGE"] >= 50), "NEW_AGE_CAT"] = "Old"

# Industry Standards: short distance: x < 1000 km, middle distance: 1000 km < x < 3000 km, long distance x > 3000 km
df.loc[(df["FLIGHT DISTANCE"] < 1000), "NEW_FLIGHT_DISTANCE_CAT"] = "Short_distance"
df.loc[(df["FLIGHT DISTANCE"] >= 1000) & (df["AGE"] < 3000), "NEW_FLIGHT_DISTANCE_CAT"] = "Middle_distance"
df.loc[(df["FLIGHT DISTANCE"] >= 3000), "NEW_FLIGHT_DISTANCE_CAT"] = "Long_distance"

df["NEW_TOTAL_SERVICE_AVG"] = df[["INFLIGHT WIFI SERVICE", "FOOD AND DRINK", "INFLIGHT ENTERTAINMENT", "INFLIGHT SERVICE", "ON-BOARD SERVICE"]].mean(axis=1)

df["NEW_TOTAL_COMFORT_AVG"] = df[["SEAT COMFORT", "LEG ROOM SERVICE", "CLEANLINESS"]].mean(axis=1)

df["NEW_TOTAL_OUTSIDEPLANE_AVG"] = df[["EASE OF ONLINE BOOKING", "GATE LOCATION", "ONLINE BOARDING", "BAGGAGE HANDLING", "CHECKIN SERVICE"]].mean(axis=1)

df["NEW_TOTAL_DELAY"] = df[["DEPARTURE DELAY IN MINUTES", "ARRIVAL DELAY IN MINUTES"]].sum(axis=1)

df["NEW_GENDER_PLUS_TRAVEL"] = df["GENDER"] + "-" + df["TYPE OF TRAVEL"]

# We created new variables so we need to use grap_col_ names again

cat_cols, num_cols, cat_but_car = grab_col_names(df)

##################################
# Encoding
##################################

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

# Update cat_cols

cat_cols = [col for col in cat_cols if df[col].dtypes == "O"]

df = one_hot_encoder(df, cat_cols, drop_first=True)

##################################
# STANDARDIZATION
##################################

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

##################################
# Data Prep for Pipeline
##################################

df = pd.read_csv("datasets/train.csv")

df.head()

def airline_data_prep(dataframe):
    dataframe = dataframe.drop("id", axis=1)
    dataframe = dataframe.drop("Unnamed: 0", axis=1)  # We don't need those variables.

    dataframe.columns = [col.upper() for col in dataframe.columns]

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    cat_cols = [col for col in cat_cols if col not in ["SATISFACTION"]]  # Satisfaciton is our target value.

    dataframe = label_encoder(dataframe, "SATISFACTION")  ###  ! (1: satisfaction) & (0: neutral or dissatisfied)

    dataframe["ARRIVAL DELAY IN MINUTES"].fillna(dataframe["DEPARTURE DELAY IN MINUTES"], inplace=True)

    for col in num_cols:

        if check_outlier(dataframe, col):
            replace_with_thresholds(dataframe, col)

    dataframe.loc[(dataframe["AGE"] < 18), "NEW_AGE_CAT"] = "Child"
    dataframe.loc[(dataframe["AGE"] >= 18) & (dataframe["AGE"] < 35), "NEW_AGE_CAT"] = "mature"
    dataframe.loc[(dataframe["AGE"] >= 35) & (dataframe["AGE"] < 50), "NEW_AGE_CAT"] = "Senior"
    dataframe.loc[(dataframe["AGE"] >= 50), "NEW_AGE_CAT"] = "Old"

    # Industry Standards: short distance: x < 1000 km, middle distance: 1000 km < x < 3000 km, long distance x > 3000 km
    dataframe.loc[(dataframe["FLIGHT DISTANCE"] < 1000), "NEW_FLIGHT_DISTANCE_CAT"] = "Short_distance"
    dataframe.loc[(dataframe["FLIGHT DISTANCE"] >= 1000) & (dataframe["AGE"] < 3000), "NEW_FLIGHT_DISTANCE_CAT"] = "Middle_distance"
    dataframe.loc[(dataframe["FLIGHT DISTANCE"] >= 3000), "NEW_FLIGHT_DISTANCE_CAT"] = "Long_distance"

    dataframe["NEW_TOTAL_SERVICE_AVG"] = dataframe[["INFLIGHT WIFI SERVICE", "FOOD AND DRINK", "INFLIGHT ENTERTAINMENT", "INFLIGHT SERVICE", "ON-BOARD SERVICE"]].mean(axis=1)

    dataframe["NEW_TOTAL_COMFORT_AVG"] = dataframe[["SEAT COMFORT", "LEG ROOM SERVICE", "CLEANLINESS"]].mean(axis=1)

    dataframe["NEW_TOTAL_OUTSIDEPLANE_AVG"] = dataframe[["EASE OF ONLINE BOOKING", "GATE LOCATION", "ONLINE BOARDING", "BAGGAGE HANDLING", "CHECKIN SERVICE"]].mean(axis=1)

    dataframe["NEW_TOTAL_DELAY"] = dataframe[["DEPARTURE DELAY IN MINUTES", "ARRIVAL DELAY IN MINUTES"]].sum(axis=1)

    dataframe["NEW_GENDER_PLUS_TRAVEL"] = dataframe["GENDER"] + "-" + dataframe["TYPE OF TRAVEL"]

    # We created new variables so we need to use grap_col_ names again

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    binary_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and dataframe[col].nunique() == 2]

    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    # Update cat_cols

    cat_cols = [col for col in cat_cols if dataframe[col].dtypes == "O"]

    dataframe = one_hot_encoder(dataframe, cat_cols, drop_first=True)

    scaler = StandardScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

    Y = dataframe["SATISFACTION"]
    X = dataframe.drop(["SATISFACTION"], axis=1)

    print("Data Set is ready to use in any model. (X = All variables), (Y = SATISFACTION)")
    return X, Y




###########################################################
##########               STANDARDIZATION             ######
###########################################################


X, y = airline_data_prep(df)

base_models(X,y,scoring="f1") #Base Model for f1_Score

base_models(X,y,scoring="roc_auc") #creating base_model for roc_auc score

base_models(X,y,scoring="accuracy") #creating base models for accuracy score

# # Focusing on 3 Models which Have Best Performance between Base Models
#     1.XGBoost
#     2.LightGBM
#     3.CatBoost

xgb=XGBClassifier()
lightgbm=LGBMClassifier()
cat=CatBoostClassifier()

###############################
# Hyperparameter Optimization
###############################

lightgbm_params={"learning_rate":[0.1,0.01,0.5],
                 "max_depth":[5,8,10],
                 "n_estimators":[75,100,500],
                 "colsample_bytree":[None,0.5,0.75,1]}

#Hyperparameter Optimization for CatBoost
cat_params={"learning_rate":[0.1,0.01,0.5],
            "depth":[5,8,10],
            "iterations":[75,100,500]}

#Hyperparameter Optimization for XGBoost
xg_params={"learning_rate":[0.1,0.01],
           "max_depth":[5,8],
           "n_estimators":[100,500,1000],
           "colsample_bytree":[None,0.5,1]}


#Final Model of LightGBM with best hyperparameters
lightgbm_grid=GridSearchCV(lightgbm,lightgbm_params,cv=3,n_jobs=-1,verbose=1).fit(X,y) #Hyperparameter Optimization for LightGBM
lightgbm_grid.best_params_
lightgbm_final=lightgbm.set_params(**lightgbm_grid.best_params_)

#Final Model of CatBoost
cat_grid=GridSearchCV(cat,cat_params,cv=3,n_jobs=-1,verbose=1).fit(X,y)
cat_final=cat.set_params(**cat_grid.best_params_)

#Final Model of XGBoost
xgb_grid=GridSearchCV(xgb,xg_params,cv=3,n_jobs=-1,verbose=1).fit(X,y)
xgb_final=xgb.set_params(**xgb_grid.best_params_)


#### Test with test data
###############################

test=pd.read_csv("test.csv")
df_test = pd.read_csv("test.csv")

X_t, y_t = airline_data_prep(df_test)

lightgbm_grid=GridSearchCV(lightgbm,lightgbm_params,cv=3,n_jobs=-1,verbose=1).fit(X_t,y_t) #Hyperparameter Optimization for LightGBM
lightgbm_grid.best_params_
lightgbm_final=lightgbm.set_params(**lightgbm_grid.best_params_)

#Final Model of CatBoost
cat_grid=GridSearchCV(cat,cat_params,cv=3,n_jobs=-1,verbose=1).fit(X_t,y_t)
cat_final=cat.set_params(**cat_grid.best_params_)

#Final Model of XGBoost
xgb_grid=GridSearchCV(xgb,xg_params,cv=3,n_jobs=-1,verbose=1).fit(X_t,y_t)
xgb_final=xgb.set_params(**xgb_grid.best_params_)


#### Plot Importance
###############################

len(xgb_final.feature_importances_)
len(df_test.drop("SATISFACTION",axis=1).columns)

plot_importance(xgb_final,df_test.drop("SATISFACTION",axis=1),num=df_test.shape[0],save=True,name="xgb")


lightgbm_final.fit(X_t,y_t)

plot_importance(lightgbm_final,df.drop("SATISFACTION",axis=1),num=df.shape[0],save=True,name="light")


cat_final.fit(X_t,y_t)

plot_importance(cat_final,df.drop("SATISFACTION",axis=1),num=df.shape[0],save=True,name="cat")



