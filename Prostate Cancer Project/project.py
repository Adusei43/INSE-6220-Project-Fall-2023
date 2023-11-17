import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

dataset = pd.read_csv('./Prostate_Cancer.csv')
dataset.head()
#print(dataset.shape)
#import pandas as pd
import numpy as np
import math
import os
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#import sys
#if not sys.warnoptions:
    #import warnings

#warnings.simplefilter("ignore")

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from numpy import inf

#dataset = pd.read_csv('./Prostate_Cancer.csv')
#dataset.head()
print(dataset.shape)

col_names = dataset.columns

print(col_names)
print(dataset.head())

# Remove leading and trailing whitespaces from column names
dataset.columns = dataset.columns.str.strip()

# Display the updated column names
print(dataset.columns)

# Display the count of unique values in the 'diagnosis_result' column
#print(dataset['diagnosis_result'].value_counts())



import os

venv_directory = "venv"  # Change this if you used a different name for your virtual environment

activate_script_path = os.path.join(venv_directory, "Scripts", "activate")

if os.path.exists(activate_script_path):
    print(f"Virtual environment '{venv_directory}' found.")
else:
    print(f"No virtual environment found for this project.")
