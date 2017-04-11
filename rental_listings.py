"""
This module predict the interest level for real estate listings.
More information on the requirements on kaggle: https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries

Untuned GBM is being used. More accurate predicitons can be created
using tuned GBM.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier as GBM
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression


def remove_outliers(df):
    '''
    All the listings with location z-score > 3 are excluded recursively

    :param df (pd.DataFrame): Dataframe with the necessary data
    :return pd.DataFrame: Dataframe with the outliers removed
    '''

    print('Length before removing outiers:', len(df))
    # Plot latitude and longtitude
    plt.scatter(df['longitude'], df['latitude'])
    plt.title('Location of the apartments before removing outliers')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    #plt.show()

    # Remove outliers that are above 3 std dev away
    for i in ['latitude', 'longitude']:
        while True:
            median = df[i].median()
            outliers = abs(df[i] - median) > 3 * df[i].std()
            if np.sum(outliers) == 0:  # No more outliers
                break
            df.loc[outliers, i] = np.nan  # exclude outliers

    df = df[df['latitude'].notnull()]
    df = df[df['longitude'].notnull()]

    print('Length after removing outiers:', len(df))
    # Plot latitude and longtitude
    plt.scatter(df['longitude'], df['latitude'])
    plt.title('Location of the apartments after removing outliers')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    #plt.show()

    return df


def create_features(df):
    '''
    Feature engineering: creating new features from existing ones

    :param df (pd.DataFrame): Dataframe with the necessary data
    :return pd.DataFrame: Dataframe with the newly created columns included
    '''

    # function to replace a string, remove leading and trailing space and make it lowercase
    func = lambda s: s.replace("\u00a0", "").strip().lower()
    fmt = lambda feat: [s.replace("\u00a0", "").strip().lower().replace(" ", "_") for s in feat]

    df["features"] = df["features"].apply(fmt)
    df["photo_count"] = df["photos"].apply(len)
    df["street_address"] = df['street_address'].apply(func)
    df["display_address"] = df["display_address"].apply(func)
    df["desc_wordcount"] = df["description"].apply(len)
    df["feature_count"] = df['features'].apply(len)
    df["bedBathDiff"] = df['bedrooms'] - df['bathrooms']
    df["bedBathSum"] = df["bedrooms"] + df['bathrooms']
    df["pricePerBed"] = df['price'] / df['bedrooms']
    df["pricePerBath"] = df['price'] / df['bathrooms']
    df["pricePerRoom"] = df['price'] / df["bedBathSum"]
    df["bedPerBath"] = df['bedrooms'] / df['bathrooms']
    df["bedsPerc"] = df["bedrooms"] / df["bedBathSum"]
    df["bathsPerc"] = df["bathrooms"] / df["bedBathSum"]

    df = df.fillna(-1).replace(np.inf, -1)
    return df


def run_GBM(X_train, Y_train, X_test):
    '''
    Run the GBM model on the given data

    :param X_train (pd.DataFrame): Training set
    :param Y_train (pd.Series): Known response to the training set
    :param X_test (pd.DataFrame): Testing set
    :return np.array: predicted probabilites associated with the testing set
    : return np.array: feature importance of the features in the training/testing set
    '''

    reg = GBM(max_features='auto', n_estimators=200, random_state=1)
    # GBM without much tuning

    reg.fit(X_train, Y_train)
    pred = reg.predict_proba(X_test)
    imp = reg.feature_importances_
    return pred, imp


def run_log(X_train, Y_train, X_test):
    '''
    Run the Log reg model on the given data.
    Performs worse than GBM

    :param X_train (pd.DataFrame): Training set
    :param Y_train (pd.Series): Known response to the training set
    :param X_test (pd.DataFrame): Testing set
    :return np.array: predicted probabilites associated with the testing set
    : return np.array: feature importance of the features in the training/testing set
    '''

    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    pred = logreg.predict_proba(X_test)
    return pred


def main(testing=False):
    '''
    Main function that runs the algorithm

    :param testing (Bool): Optional paramter for testing purposes
    :return: None
    '''

    # Load data
    training_set = pd.read_json("train.json").sort_values(by="listing_id")
    testing_set = pd.read_json("test.json").sort_values(by="listing_id")

    training_set = remove_outliers(training_set)
    Y_train = training_set['interest_level']

    X_train = create_features(training_set)
    X_test = create_features(testing_set)
    # There can be more feature engineering done on the
    # manager_ids, building_ids, certain words appearing in the description,
    # as well as type of amenties

    del X_train['interest_level']

    # Convert the interest into numeric form
    labels = {'high': 0, 'medium': 1, 'low': 2}
    Y_train = Y_train.apply(lambda x: labels[x])

    if testing:
        # features cols listed in order of importance
        # from most important to least important
        feature_cols = ['latitude', 'price', 'longitude',
                        'pricePerRoom', 'desc_wordcount', 'feature_count',
                        'pricePerBed', 'photo_count', 'pricePerBath',
                        'bedBathDiff', 'bedPerBath', 'bedrooms',
                        'bathrooms', 'bedBathSum', 'bathsPerc', 'bedsPerc']

        # Excluding the last 4 columns gives the best performance
        feature_cols = ['latitude', 'price', 'longitude',
                        'pricePerRoom', 'desc_wordcount', 'feature_count',
                        'pricePerBed', 'photo_count', 'pricePerBath',
                        'bedBathDiff', 'bedPerBath', 'bedrooms']

        X_train = X_train[feature_cols]
        X_test = X_test[feature_cols]

        cv_scores_GBM = []
        importances_GBM = []
        cv_scores_log = []

        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

        for dev_index, val_index in kf.split(X_train, Y_train):
            train_X_X = X_train[X_train.index.isin(dev_index)]
            test_X_X = X_train[X_train.index.isin(val_index)]
            train_Y_Y = Y_train[Y_train.index.isin(dev_index)]
            test_Y_Y = Y_train[Y_train.index.isin(val_index)]

            pred_GBM, imp = run_GBM(train_X_X, train_Y_Y, test_X_X)
            pred_log = run_log(train_X_X, train_Y_Y, test_X_X)
            cv_scores_GBM.append(log_loss(test_Y_Y, pred_GBM))
            importances_GBM.append(imp)
            cv_scores_log.append(log_loss(test_Y_Y, pred_log))

        print(np.mean(cv_scores_GBM))
        print(np.mean(cv_scores_log))
        print(importances_GBM)

    feature_cols = ['latitude', 'price', 'longitude',
                    'pricePerRoom', 'desc_wordcount', 'feature_count',
                    'pricePerBed', 'photo_count', 'pricePerBath',
                    'bedBathDiff', 'bedPerBath', 'bedrooms']

    X_train = X_train[feature_cols]
    X_test = X_test[feature_cols]

    pred_GBM, imp = run_GBM(X_train, Y_train, X_test)

    X_test_with_prob = pd.DataFrame({'listing_id': testing_set['listing_id'],
                                     'high': pred_GBM[:, 0],
                                     'medium': pred_GBM[:, 1],
                                     'low': pred_GBM[:, 2]})

    # Enforce the order
    X_test_with_prob = X_test_with_prob[['listing_id', 'high',
                                        'medium', 'low']
    X_test_with_prob.to_csv('submission.csv', index=False)

    '''
    This submission scored a log loss of 0.63135 on kaggle's public leaderboard.
    I will continue to work on my code to improve the score.

    I am not 100% sure if thats an accurate number or not because csv files
    have a limit of 65536 rows but the testing dataset has 74659 rows.
    '''

if __name__ == "__main__":
    main()
