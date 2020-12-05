# -*- coding: utf-8 -*-

import pandas as pd
import sklearn
from sklearn import preprocessing

# for data visualizations
import matplotlib.pyplot as plt
import seaborn as sns

categorical_labels = ["Month", "Weekend", "Revenue", "VisitorType"]


def get_data_cleaned():
    data_frame = read_data()
    data_size = len(data_frame)

    encode_label(data_frame)
    data_frame_scaled = standardize(data_frame)
    remove_outliers(data_frame_scaled)
    draw_correlation(data_frame_scaled)

    data_size_2 = len(data_frame_scaled)
    size_difference = data_size - data_size_2

    return data_frame_scaled


def read_data(file_name='online_shoppers_intention.csv'):
    data_frame = pd.read_csv(file_name)
    # data.isnull().sum().values
    # remove any nulls
    data_frame = data_frame.dropna()

    # Dropping the negative Durations
    data_frame = data_frame.drop(data_frame[data_frame['Administrative_Duration'] < 0].index)
    data_frame = data_frame.drop(data_frame[data_frame['Informational_Duration'] < 0].index)
    data_frame = data_frame.drop(data_frame[data_frame['ProductRelated_Duration'] < 0].index)
    # Checking , no negative values
    data_frame.describe()

    return data_frame


def encode_label(data_frame):

    label_encode = sklearn.preprocessing.LabelEncoder()

    for label in categorical_labels:
        data_frame[label] = label_encode.fit_transform(data_frame[label])

    data_frame[categorical_labels].head(11)


def standardize(data_frame):
    # Standardization Standardization involves centering the variable at zero, and standardizing the variance to 1.
    # The procedure involves subtracting the mean of each observation and then dividing by the standard deviation: z
    # = (x - x_mean) / std

    # the scaler - for standardization

    # standardisation: with the StandardScaler from sklearn

    # set up the scaler
    scaler = sklearn.preprocessing.StandardScaler()

    # fit the scaler to the train set, it will learn the parameters
    scaler.fit(data_frame)
    _data_scaled = scaler.transform(data_frame)
    data_scaled = pd.DataFrame(_data_scaled, columns=data_frame.columns)
    # data_scaled[categorical_labels + ["Administrative_Duration"]].head(11)

    # restore the categorical values because we do not standardize these
    for label in categorical_labels:
        data_scaled[label] = data_frame[label].to_numpy()

    #data_scaled[categorical_labels + ["Administrative_Duration"]].head(11)

    # test if is a bug in the library
    var = data_scaled.isna().sum().values

    return data_scaled


def compare_scaling(data_frame, data_frame_scaled):
    # let's compare the variable distributions before and after scaling
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(22, 5))
    ax1.set_xlim([-120, 600])
    ax1.set_ylim([0, 0.017])
    ax2.set_xlim([-1.2, 8])
    ax2.set_ylim([0, 2.5])

    # before scaling
    ax1.set_title('Before Scaling')
    sns.kdeplot(data_frame['Administrative_Duration'], ax=ax1)
    sns.kdeplot(data_frame['Informational_Duration'], ax=ax1)
    sns.kdeplot(data_frame['ProductRelated_Duration'], ax=ax1)

    # after scaling
    ax2.set_title('After Standard Scaling')
    sns.kdeplot(data_frame_scaled['Administrative_Duration'], ax=ax2)
    sns.kdeplot(data_frame_scaled['Informational_Duration'], ax=ax2)
    sns.kdeplot(data_frame_scaled['ProductRelated_Duration'], ax=ax2)


def draw_boxplots(data_frame_scaled):
    plt.rcParams['figure.figsize'] = (40, 35)
    plt.subplot(3, 3, 1)
    sns.set_theme(style="whitegrid")
    # sns.boxplot(data = data_scaled,palette="Set3", linewidth=2.5)
    sns.boxenplot(data=data_frame_scaled, orient="h", palette="Set3")
    # sns.stripplot(data=data,orient="h",size=4, color=".26")

    plt.title('box plots types', fontsize=10)


# ------------------------------------------------------------------------------
# accept a dataframe, remove outliers, return cleaned data in a new dataframe
# see http://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm
# ------------------------------------------------------------------------------
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3 - q1  # Interquartile range
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


def remove_outliers(data_frame_scaled):
    for col in data_frame_scaled.columns:
        Q1 = data_frame_scaled[col].quantile(0.25)
        Q3 = data_frame_scaled[col].quantile(0.75)
        IQR = Q3 - Q1  # IQR is interquartile range.
        #print(IQR)
        filter = (data_frame_scaled[col] > Q1 - 1.5 * IQR) & (data_frame_scaled[col] < Q3 + 1.5 * IQR)
        data_frame_scaled = data_frame_scaled.loc[filter]


def draw_correlation(data_frame_scaled):
    plt.figure(figsize=(20, 15))
    ax = sns.heatmap(data_frame_scaled.corr(), cmap='Blues', linecolor='Black', linewidths=.3, annot=True, fmt=".3")
    ax.set_title('The Correlation Heatmap')
    bottom, top = ax.get_ylim()

