
# Previous version:     api_v3.py
# This version:         api_v6.py
#
# Changes made in this version are as follows:
#   1) All harcoded path joining modified with os.path.join()
#   2) Now, the selected model only will get published (when published by user), not the latest model.
#   3) Now, only the list of published model will show up in the test tab.
#   4) global train_df, global test_df, and global model eliminated.
#   5) global train_flag, global test_flag, and global scale_flag eliminated
#   6) metadata_main_format and metadata_model_dict updated
#   7) os.getcwd() replaced with get_project_folder_path(username, projectname)
#   8) additions to get_toolkit_dir(), get_clients_dir(), get_projects_folder_path(username), get_project_folder_path(username, projectname), get_user_folder_path(username)
#   9) Metadata loading & dumping now done using functions
#  10) os.chdir() now eliminated 


print('\n ===== START =====\n')

from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import os
import csv
import json
#import datetime
#from decimal import Decimal
#import matplotlib.pyplot as plt
#import seaborn as sns
import plotly
import plotly.express as px
#from plotly.offline import plot
#import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

#user_json_path = 'C:\\Users\\DELL\\Toolkit' # Change if needed
#user_json_filename = 'jct.json' # Change if needed
#client_folder_path='C:\\Users\\DELL\\Toolkit\\Clients' # Change if needed
'''
metadata_main_format = {
                "project_description": 0,
                "input":{
                            "training_set":0,
                            "train_flag":0,
                            "testing_set":0,
                            "test_flag":0
                        },

                "clean_train_set":{
                                    "drop":0,
                                    "replace":0,
                                    "fill":0,
                                    "convert":0,
                                    "encode":0
                                  },
                "clean_test_set":{
                                    "drop":0,
                                    "replace":0,
                                    "fill":0,
                                    "convert":0,
                                    "encode":0
                                 },
                "model_id_counter":0,
                "model":[],
                "import":0,
                "test":0
              }
'''
#train_df = None
#train_flag = None
#test_df = None
#test_flag = None
#model = None
#scale_flag = 0

# os.path.join(path, "Downloads", "file.txt", "/home")

# ========== START (Non-API functions & classes) ==========

'''
def user_json_priming():
    # os.path.exists(user_json_path+'\\'+user_json_filename)
    if os.path.exists(os.path.join(user_json_path, user_json_filename)):
        print('\n ===== User JSON present =====\n')
        pass
    else:
        print('\n ===== Creating new user JSON =====\n')
        main_format = [
            {
        		"username": "xyz",
        		"project_id_counter": 0,
        		"project_details": []
        	},
            {
            	"username": "abc",
            	"project_id_counter": 0,
            	"project_details": []
            },
            {
            	"username": "pqr",
            	"project_id_counter": 0,
            	"project_details": []
            }]

        #DUMP JSON
        with open(os.path.join(user_json_path, user_json_filename), 'w') as file:
            json.dump(main_format, file, indent=4)
        print('\n ===== New user JSON created =====\n')

def read_csv_file(csv_file): # To support Front-end tabular form
    """
    Read CSV file
    Args:
        csv_file: file name

    Returns:
        Array of dict

    """
    try:
        from decimal import Decimal
        array_of_dict = []
        # print('Reading csv file:' + csv_file)
        with open(csv_file, 'r') as csv_file:
            for line in csv.DictReader(csv_file):
                for item in line:
                    if line[item] is None or line[item] == '' or line[item] == 'NA':
                        line[item] = 'NA'
                    try:
                        line[item] = round(float(line[item]), 2)
                    except:
                        line[item] = line[item]
                array_of_dict.append(line)
        return array_of_dict
    except IOError:
        raise ValueError('Couldn\'t find csv file:' + csv_file)
'''

# ========== Fuctions from Anand Adake Sir: START ========== #

def read_csv_file(csv_file):
    """
    Read CSV file
    Args:
        csv_file: file name

    Returns:
        Array of dict

    """
    array_of_dict = []
    try:
        from decimal import Decimal
        with open(csv_file, 'r') as csv_file:
            for line in csv.DictReader(csv_file):
                for item in line:
                    if line[item] is None or line[item] is '' or line[item] is 'NA':
                        line[item] = 'NA'
                    try:
                        line[item] = round(float(line[item]), 2)
                    except:
                        line[item] = line[item]
                array_of_dict.append(line)
    except IOError:
        raise ValueError('Could not find csv file:' + csv_file)
    return array_of_dict

def create_dir(dir_path):
    """
    Create the directory.
    Args:
        dir_path: directory path

    Returns:
        None

    """
    if os.path.exists(dir_path):
        if os.path.isdir(dir_path):
            # empty_dir(dir_path)
            os.chmod(dir_path, 0o777)  # change the directory permission (read,write,execute [777]).
    else:
        os.mkdir(dir_path)

def create_nested_dirs(base_path, req_dir_path):
    """
        Create nested directories.
        Args:
            base_path: base directory path
            req_dir_path: list of directories

        Returns:
            last nested directory path.

        """
    # change the directory permission (read,write,execute [777]).
    os.chmod(base_path, 0o777)
    dir_path = ''
    for index, dirs in enumerate(req_dir_path, 1):
        dir_path = os.path.abspath(os.path.join(base_path, os.sep.join(req_dir_path[:index])))
        if not os.path.exists(dir_path):
            create_dir(dir_path)
    return dir_path

def copy_dir(src_dir, dst_dir):
    """
    Copy content from source directory to destination directory
    Args:
        src_dir: source directory path
        dst_dir: destination directory path

    Returns:
        None

    """
    from distutils.dir_util import copy_tree
    copy_tree(src_dir, dst_dir)

    # change the directory permission (read,write,execute [777]). It is needed for directory rename to date
    os.chmod(dst_dir, 0o777)

def empty_dir(dir_path):
    """
    Delete all files and sub directories from a directory
    Args:
        dir_path: directory path

    Returns:
        None

    """
    import shutil
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            raise

def read_json_file(json_file):
    """
    Read JSON file
    Args:
        json_file: file name

    Returns:
        dict

    """
    try:
        # print('Reading json file:' + json_file)
        with open(json_file) as file_path:
            return json.load(file_path)
    except IOError:
        print('Could not find json file:' + json_file)
        raise

def write_json_file(json_content, json_file):
    """
    Write json_content into the file.
    Args:
        json_content: dict
        json_file: file Name

    Returns:
        None

    """
    with open(json_file, 'w') as file_path:
        json.dump(json_content, file_path, indent=4, default=pandas_datatype_converter)

def pandas_datatype_converter(obj):
    """
    Convert pandas data type to python data type.
    It is mainly used to remove TypeError: Object of type 'pandas data type e.g. int64' is not JSON serializable
    Args:
        obj: This is automatically sent during json.dump call

    Returns:
        Converted python data type from pandas data type

    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.__str__()

def read_pickle(pickle_file):
    # infile = open(pickle_file, 'rb')
    # model = pickle.load(infile)
    # infile.close()
    # return model
    return pickle.load(pickle_file) # pd.read_pickle(pickle_file)

def write_pickle(data, pickle_file):
    file = open(pickle_file, "wb")
    pickle.dump(data, file)
    file.close()

# -------------------------------------------------------------------------- #

def get_metadata_main_format():
    return {
                    "project_description": 0,
                    "input":{
                                "training_set":0,
                                "train_flag":0,
                                "testing_set":0,
                                "test_flag":0
                            },

                    "clean_train_set":{
                                        "drop":0,
                                        "replace":0,
                                        "fill":0,
                                        "convert":0,
                                        "encode":0
                                      },
                    "clean_test_set":{
                                        "drop":0,
                                        "replace":0,
                                        "fill":0,
                                        "convert":0,
                                        "encode":0
                                     },
                    "model_id_counter":0,
                    "model":[],
                    "import":0,
                    "test":0
                  }

def get_toolkit_dir():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    toolkit_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'Toolkit'))
    if os.path.exists(toolkit_dir):
        print('\n === TD EXISTS ===\n')
    else:
        print('\n === TD NO EXIST ===\n')
        os.mkdir(toolkit_dir)
        print('\n === TD CREATED NOW ===\n')
    return toolkit_dir

def get_clients_dir():
    clients_dir = os.path.abspath(os.path.join(get_toolkit_dir(), 'Clients'))
    if os.path.exists(clients_dir):
        print('\n === CD EXISTS ===\n')
    else:
        print('\n === CD NO EXIST ===\n')
        os.mkdir(clients_dir)
        print('\n === CD CREATED NOW ===\n')
    return clients_dir

def get_user_folder_path(username):
    user_folder_path = os.path.abspath(os.path.join(get_clients_dir(), username))
    if os.path.exists(user_folder_path):
        print('\n === UD EXISTS ===\n')
    else:
        print('\n === UD NO EXIST ===\n')
        os.mkdir(user_folder_path)
        print('\n === UD CREATED NOW ===\n')
    return user_folder_path

def get_projects_folder_path(username):
    projects_folder_path =  os.path.join(get_user_folder_path(username), 'Projects')
    if os.path.exists(projects_folder_path):
        print('\n === PsD EXISTS ===\n')
    else:
        print('\n === PsD NO EXIST ===\n')
        os.mkdir(projects_folder_path)
        print('\n === PsD CREATED NOW ===\n')
    return projects_folder_path

def get_project_folder_path(username, projectname):
    project_folder_path = os.path.join(get_projects_folder_path(username), projectname)
    if os.path.exists(project_folder_path):
        print('\n === PD EXISTS ===\n')
    else:
        print('\n === PD NO EXIST ===\n')
        os.mkdir(project_folder_path)
        print('\n === PD CREATED NOW ===\n')
    return project_folder_path

def get_user_json_filename():
    return 'jct.json' # Change if needed

def get_metadata_json_path(username,projectname):
    return os.path.abspath(os.path.join(get_project_folder_path(username,projectname), 'Metadata.json'))

def get_user_json_file_path():
    return os.path.abspath(os.path.join(get_toolkit_dir(), get_user_json_filename()))

def user_json_priming():
    if not os.path.exists(get_user_json_file_path()):
        print('\n ===== Creating new user JSON =====\n')
        main_format = [
        {
        "username": "xyz",
        "project_id_counter": 0,
        "project_details": []
        },
        {
        "username": "abc",
        "project_id_counter": 0,
        "project_details": []
        },
        {
        "username": "pqr",
        "project_id_counter": 0,
        "project_details": []
        }]
        write_json_file(main_format, get_user_json_file_path())
        print('\n ===== New user JSON created =====\n')
user_json_priming() # Priming the user JSON (useful for first run only)


# ========== Functions from Anand Adake sir: END ========== #

'''
##### DISCONTINUED; Can un-comment later, if required #####
class DecimalEncoder (json.JSONEncoder): # To convert Decimal objects into float (JSON serializable)
    def default (self, obj):
       if isinstance (obj, Decimal):
           return float (obj)
       return json.JSONEncoder.default (self, obj)
'''

def target_selection(df, target_col):
    y=df[target_col]
    X=df.drop(target_col, axis=1)
    return (X, y)

def TVS_yes(X, y, rss):
    from sklearn.model_selection import train_test_split
    if rss=='no':
        X_train, X_val, y_train, y_val=train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    else:
        X_train, X_val, y_train, y_val=train_test_split(X, y, test_size=0.3, stratify=y)
    return (X_train, X_val, y_train, y_val)

def TVS_no(X, y):
    X_train=X.copy()
    y_train=y.copy()
    X_val=None
    y_val=None
    return (X_train, X_val, y_train, y_val)

def normalize(username, projectname, tvs_flag, X_train, X_val, model_id):
    from sklearn.preprocessing import MinMaxScaler
    norm=MinMaxScaler()

    if tvs_flag==1:
        try:
            X_train_norm = pd.DataFrame(norm.fit_transform(X_train))
        except:
            return (None, None, None)

        # saving this scaler here so that we can apply it to test set later
        import pickle
        #pkl_file = get_project_folder_path(username, projectname)+ '\\Training folder\\scaler_model_id_{}.pkl'.format(model_id)
        pkl_file = os.path.join(get_project_folder_path(username, projectname), "Training folder", "scaler_model_id_{}.pkl".format(model_id))
        data = norm # saving the scaler file
        file = open(pkl_file,"wb")
        pickle.dump(data,file)
        file.close()

        X_val_norm = pd.DataFrame(norm.transform(X_val))
        X_train_norm.columns = X_train.columns
        X_val_norm.columns = X_val.columns
        scale_flag='n1'

    else:
        try:
            X_train_norm = pd.DataFrame(norm.fit_transform(X_train))
        except:
            return (None, None, None)

        import pickle
        #pkl_file = get_project_folder_path(username, projectname)+ '\\Training folder\\scaler_model_id_{}.pkl'.format(model_id)
        pkl_file = os.path.join(get_project_folder_path(username, projectname), "Training folder", "scaler_model_id_{}.pkl".format(model_id))
        data = norm
        file = open(pkl_file,"wb")
        pickle.dump(data,file)
        file.close()

        X_val_norm = None
        X_train_norm.columns = X_train.columns
        scale_flag='n0'
    return (X_train_norm, X_val_norm, scale_flag)

def standardize(username, projectname, tvs_flag, X_train, X_val, model_id):
    from sklearn.preprocessing import StandardScaler
    norm=StandardScaler()

    if tvs_flag==1:
        try:
            X_train_norm = pd.DataFrame(norm.fit_transform(X_train))
        except:
            return (None, None, None)

        import pickle
        #pkl_file = get_project_folder_path(username, projectname)+ '\\Training folder\\scaler_model_id_{}.pkl'.format(model_id)
        pkl_file = os.path.join(get_project_folder_path(username, projectname), "Training folder", "scaler_model_id_{}.pkl".format(model_id))
        data = norm # saving the scaler file
        file = open(pkl_file,"wb")
        pickle.dump(data,file)
        file.close()

        X_val_norm = pd.DataFrame(norm.transform(X_val))
        X_train_norm.columns = X_train.columns
        X_val_norm.columns = X_val.columns
        scale_flag='s1'

    else:
        try:
            X_train_norm = pd.DataFrame(norm.fit_transform(X_train))
        except:
            return (None, None, None)

        import pickle
        #pkl_file = get_project_folder_path(username, projectname)+ '\\Training folder\\scaler_model_id_{}.pkl'.format(model_id)
        pkl_file = os.path.join(get_project_folder_path(username, projectname), "Training folder", "scaler_model_id_{}.pkl".format(model_id))
        data = norm
        file = open(pkl_file,"wb")
        pickle.dump(data,file)
        file.close()

        X_val_norm = None
        X_train_norm.columns = X_train.columns
        scale_flag='s0'
    return (X_train_norm, X_val_norm, scale_flag)

def fit_model(algo, X_train, y_train, tud):
    print('\n Primary fitting IN')
    if algo==None:
        print('\n Algorithm None')
        return (None, None)

    if algo=='Logistic Regression':
        print('\n Logistic Regression IN')
        tune_key=1
        from sklearn.linear_model import LogisticRegression
        try:
            lor=LogisticRegression(penalty=tud['penalty'], C=tud['C'], solver=tud['solver'], random_state=1)
            model=lor.fit(X_train, y_train)
        except:
            print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
            return (None, None)
        return (model, tune_key)

    if algo=='KNN Classifier':
        print('\n KNN Classifier IN')
        tune_key=2
        from sklearn.neighbors import KNeighborsClassifier as KNN
        try:
            knn=KNN(n_neighbors=tud['n_neighbors'])
            model=knn.fit(X_train, y_train)
        except:
            print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
            return (None, None)
        return (model, tune_key)

    if algo=='SVM':
        print('\n SVM IN')
        tune_key=3
        from sklearn import svm
        try:
            svc=svm.SVC(kernel=tud['kernel'], degree=tud['degree'], C=tud['C'], gamma=tud['gamma'])
            model=svc.fit(X_train, y_train)
        except:
            print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
            return (None, None)
        return (model, tune_key)

    if algo=='Decision Tree Classifier':
        print('\n DTC IN')
        tune_key=4
        from sklearn.tree import DecisionTreeClassifier
        for param in ['max_depth', 'max_leaf_nodes']:
            if tud[param]=='None':
                tud[param]=None
        print('\n\n ===== TUD =====\n\n', tud)
        try:
            dt=DecisionTreeClassifier(max_depth=tud['max_depth'], min_samples_leaf=tud['min_samples_leaf'], min_samples_split=tud['min_samples_split'], max_leaf_nodes=tud['max_leaf_nodes'], random_state=1)
            model=dt.fit(X_train, y_train)
        except:
            print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
            return (None, None)
        return (model, tune_key)

    if algo=='Random Forest Classifier':
        print('\n RFC IN')
        tune_key=5
        from sklearn.ensemble import RandomForestClassifier
        for param in ['max_depth', 'max_leaf_nodes']:
            if tud[param]=='None':
                tud[param]=None
        try:
            rf=RandomForestClassifier(n_estimators=tud['n_estimators'], max_depth=tud['max_depth'], min_samples_leaf=tud['min_samples_leaf'], min_samples_split=tud['min_samples_split'], max_leaf_nodes=tud['max_leaf_nodes'], random_state=1)
            model=rf.fit(X_train, y_train)
        except:
            print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
            return (None, None)
        return (model, tune_key)

    if algo=='Linear Regression':
        print('\n Linear Regression IN')
        tune_key=6
        from sklearn.linear_model import LinearRegression
        lir=LinearRegression()
        try:
            model=lir.fit(X_train, y_train)
        except:
            print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
            return (None, None)
        return (model, tune_key)

    if algo=='KNN Regressor':
        print('\n KNN Regression IN')
        tune_key=7
        from sklearn.neighbors import KNeighborsRegressor as KNN
        try:
            knn=KNN(n_neighbors=tud['n_neighbors'])
            model=knn.fit(X_train, y_train)
        except:
            print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
            return (None, None)
        return (model, tune_key)

    if algo=='SVR':
        print('\n SVR IN')
        tune_key=8
        from sklearn.svm import SVR
        try:
            svr=SVR(kernel=tud['kernel'], epsilon=tud['epsilon'], degree=tud['degree'], C=tud['C'], gamma=tud['gamma'])
            model=svr.fit(X_train, y_train)
        except:
            print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
            return (None, None)
        return (model, tune_key)

    if algo=='Decision Tree Regressor':
        print('\n DTR IN')
        tune_key=9
        from sklearn.tree import DecisionTreeRegressor
        for param in ['max_depth', 'max_leaf_nodes']:
            if tud[param]=='None':
                tud[param]=None
        try:
            dt=DecisionTreeRegressor(max_depth=tud['max_depth'], min_samples_leaf=tud['min_samples_leaf'], min_samples_split=tud['min_samples_split'], max_leaf_nodes=tud['max_leaf_nodes'], random_state=1)
            model=dt.fit(X_train, y_train)
        except:
            print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
            return (None, None)
        return (model, tune_key)

    if algo=='Random Forest Regressor':
        print('\n RFR IN')
        tune_key=10
        from sklearn.ensemble import RandomForestRegressor
        for param in ['max_depth', 'max_leaf_nodes']:
            if tud[param]=='None':
                tud[param]=None
        try:
            rf=RandomForestRegressor(n_estimators=tud['n_estimators'], max_depth=tud['max_depth'], min_samples_leaf=tud['min_samples_leaf'], min_samples_split=tud['min_samples_split'], max_leaf_nodes=tud['max_leaf_nodes'], random_state=1)
            model=rf.fit(X_train, y_train)
        except:
            print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
            return (None, None)
        return (model, tune_key)

def validation_pred(model, tvs_flag, X_train, X_val):
    if tvs_flag==1:
        pred_train=model.predict(X_train)
        pred_val=model.predict(X_val)
        pvf=1
    else:
        pred_train=model.predict(X_train)
        pred_val=None
        pvf=0
    return (pred_val, pred_train, pvf)

def training_pred_folder_creation(pred_train, X_train, y_train, train_folder_path):
    output_train=X_train.reset_index().copy()
    train_predictions=pd.DataFrame(pred_train)
    train_predictions.columns=['Predicted_target']
    train_target=y_train.reset_index(drop=True).copy()
    #val_target.columns=['Actual_target']
    output_train=pd.concat([output_train, train_predictions], axis=1)
    output_train=pd.concat([output_train, train_target], axis=1)
    #output_train.to_csv(train_folder_path+'\\Predictions_for_TrainSet.csv', index=False)
    output_train.to_csv(os.path.join(train_folder_path, "Predictions_for_TrainSet.csv"), index=False)

def validation_pred_folder_creation(pred_val, X_val, y_val, train_folder_path):
    output_val=X_val.reset_index().copy()
    val_predictions=pd.DataFrame(pred_val)
    val_predictions.columns=['Predicted_target']
    val_target=y_val.reset_index(drop=True).copy()
    #val_target.columns=['Actual_target']
    output_val=pd.concat([output_val, val_predictions], axis=1)
    output_val=pd.concat([output_val, val_target], axis=1)
    #output_val.to_csv(train_folder_path+'\\Predictions_for_ValidationSet.csv', index=False)
    output_val.to_csv(os.path.join(train_folder_path, "Predictions_for_ValidationSet.csv"), index=False)

def evaluation_metrics_setup(algo):
    if algo in ['Logistic Regression', 'KNN Classifier', 'SVM', 'Decision Tree Classifier', 'Random Forest Classifier']:
        evaluation_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC', 'Log-Loss']
    else:
        evaluation_metrics = ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Root Mean Squared Log Error (RMSLE)', 'R2', 'Adjusted R2']
    return (evaluation_metrics)

def metric_calculation(model_id, tvs_flag, evaluation_metrics, pred_val, y_val, pred_train, y_train):
    if 'Accuracy' in evaluation_metrics: # Ensuring classification case
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        if tvs_flag == 1:
            cm_train = confusion_matrix(y_train, pred_train, labels=np.unique(y_train).tolist())
            cm_val = confusion_matrix(y_val, pred_val, labels=np.unique(y_val).tolist())

            cmtdf = pd.DataFrame(cm_train, columns=np.unique(y_train).tolist(), index=np.unique(y_train).tolist())
            cmvdf = pd.DataFrame(cm_val, columns=np.unique(y_val).tolist(), index=np.unique(y_val).tolist())

            ddata=cmtdf.values.tolist()
            dcolumns=cmtdf.columns.tolist()
            dindex=cmtdf.index.tolist()
            for i, d in enumerate(ddata):
                d.insert(0, list(cmtdf.index)[i])
            dcolumns.insert(0, 'Training {}'.format(model_id))
            cmtdf = pd.DataFrame(columns=dcolumns, data=ddata)

            ddata=cmvdf.values.tolist()
            dcolumns=cmvdf.columns.tolist()
            dindex=cmvdf.index.tolist()
            for i, d in enumerate(ddata):
                d.insert(0, list(cmvdf.index)[i])
            dcolumns.insert(0, 'Validation {}'.format(model_id))
            cmvdf = pd.DataFrame(columns=dcolumns, data=ddata)

            #cmtdf.append(pd.Series(), ignore_index=True)
            #cmvdf.append(pd.Series(), ignore_index=True)

            cmdf = pd.concat([cmtdf, cmvdf], axis=0, ignore_index=True)
        else:
            cm_train = confusion_matrix(y_train, pred_train, labels=np.unique(y_train).tolist())
            cm_val = None

            cmtdf = pd.DataFrame(cm_train, columns=np.unique(y_train).tolist(), index=np.unique(y_train).tolist())
            cmvdf = None

            ddata=cmtdf.values.tolist()
            dcolumns=cmtdf.columns.tolist()
            dindex=cmtdf.index.tolist()
            for i, d in enumerate(ddata):
                d.insert(0, list(cmtdf.index)[i])
            dcolumns.insert(0, 'Training {}'.format(model_id))
            cmtdf = pd.DataFrame(columns=dcolumns, data=ddata, index=dindex)

            #cmtdf.append(pd.Series(), ignore_index=True)

            cmdf = cmtdf.copy()
    else:
        cmtdf = None
        cmvdf = None
        cmdf = None

    train_metrics = []
    val_metrics = []
    for metric in evaluation_metrics:
        if metric=='Accuracy':
            from sklearn.metrics import accuracy_score as acc
            if tvs_flag==1:
                acc_train=acc(pred_train, y_train)
                acc_val=acc(pred_val, y_val)
                print('\n Train Accuracy:', acc_train, '\n Validation Accuracy:', acc_val)
                train_metrics.append(acc_train)
                val_metrics.append(acc_val)
            else:
                acc_train=acc(pred_train, y_train)
                acc_val=None
                print('\n Train Accuracy:', acc_train, '\n Validation Accuracy:', acc_val)
                train_metrics.append(acc_train)
                val_metrics.append(acc_val)

        if metric=='Precision':

            from sklearn.metrics import precision_score as ps
            if tvs_flag==1:
                ps_train=ps(pred_train, y_train, average='weighted')
                ps_val=ps(pred_val, y_val, average='weighted')
                print('\n Train Precision:', ps_train, '\n Validation Precision:', ps_val)
                train_metrics.append(ps_train)
                val_metrics.append(ps_val)
            else:
                ps_train=ps(pred_train, y_train, average='weighted')
                ps_val=None
                print('\n Train Precision:', ps_train, '\n Validation Precision:', ps_val)
                train_metrics.append(ps_train)
                val_metrics.append(ps_val)

        if metric=='Recall':
            from sklearn.metrics import recall_score as rs
            if tvs_flag==1:
                rs_train=rs(pred_train, y_train, average='weighted')
                rs_val=rs(pred_val, y_val, average='weighted')
                print('\n Train Recall:', rs_train, '\n Validation Recall:', rs_val)
                train_metrics.append(rs_train)
                val_metrics.append(rs_val)
            else:
                rs_train=rs(pred_train, y_train, average='weighted')
                rs_val=None
                print('\n Train Recall:', rs_train, '\n Validation Recall:', rs_val)
                train_metrics.append(rs_train)
                val_metrics.append(rs_val)

        if metric=='F1-score':
            from sklearn.metrics import f1_score as fs
            if tvs_flag==1:
                try:
                    fs_train=fs(pred_train, y_train, average='weighted')
                    fs_val=fs(pred_val, y_val, average='weighted')

                except:
                    fs_train='--'
                    fs_val='--'
                print('\n F1-score for training set:', fs_train, '\n F1-score for validation set:', fs_val)
                train_metrics.append(fs_train)
                val_metrics.append(fs_val)
            else:
                try:
                    fs_train=fs(pred_train, y_train, average='weighted')
                except:
                    fs_train='--'
                fs_val=None
                print('\n F1-score for training set:', fs_train, '\n F1-score for validation set:', fs_val)
                train_metrics.append(fs_train)
                val_metrics.append(fs_val)

        if metric=='AUC-ROC':
            from sklearn.metrics import roc_auc_score as ras
            if tvs_flag==1:
                try:
                    ras_train=ras(pred_train, y_train,  average='weighted', multi_class='ovr')
                    ras_val=ras(pred_val, y_val,  average='weighted', multi_class='ovr')
                except:
                    ras_train='--'
                    ras_val='--'
                print('\n AUC-ROC for training set:', ras_train, '\n AUC-ROC for validation set:', ras_val)
                train_metrics.append(ras_train)
                val_metrics.append(ras_val)
            else:
                try:
                    ras_train=ras(pred_train, y_train,  average='weighted', multi_class='ovr')
                except:
                    ras_train='--'
                ras_val=None
                print('\n AUC-ROC for training set:', ras_train, '\n AUC-ROC for validation set:', ras_val)
                train_metrics.append(ras_train)
                val_metrics.append(ras_val)

        if metric=='Log-Loss':
            from sklearn.metrics import log_loss as ll
            if tvs_flag==1:
                try:
                    ll_train=ll(pred_train, y_train)
                    ll_val=ll(pred_val, y_val)
                except:
                    ll_train='--'
                    ll_val='--'
                print('\n Log-loss score for training set:', ll_train, '\n Log-loss score for validation set:', ll_val)
                train_metrics.append(ll_train)
                val_metrics.append(ll_val)
            else:
                ll_train=ll(pred_train, y_train)
                ll_val=None
                print('\n Log-loss score for training set:', ll_train, '\n Log-loss score for validation set:', ll_val)
                train_metrics.append(ll_train)
                val_metrics.append(ll_val)

        if metric=='True-Positive': # DISCONTINUED
            from sklearn.metrics import confusion_matrix
            if tvs_flag==1:
                tp_train, fn_train, fp_train, tn_train = confusion_matrix(pred_train, y_train).reshape(-1)
                tp_val, fn_val, fp_val, tn_val = confusion_matrix(pred_val, y_val).reshape(-1)
                print('\n TP train:', tp_train, '\n FN train:', fn_train, '\n FP train:', fp_train, '\n TN train:', tn_train)
                print('\n TP Validation:', tp_val, '\n FN Validation:', fn_val, '\n FP Validation:', fp_val, '\n TN Validation:', tn_val)
                train_metrics.append(tp_train)
                train_metrics.append(fn_train)
                train_metrics.append(fp_train)
                train_metrics.append(tn_train)
                val_metrics.append(tp_val)
                val_metrics.append(fn_val)
                val_metrics.append(fp_val)
                val_metrics.append(tn_val)
            else:
                tp_train, fn_train, fp_train, tn_train = confusion_matrix(pred_train, y_train).reshape(-1)
                tp_val, fn_val, fp_val, tn_val = (None, None, None, None)
                print('\n TP train:', tp_train, '\n FN train:', fn_train, '\n FP train:', fp_train, '\n TN train:', tn_train)
                print('\n TP Validation:', tp_val, '\n FN Validation:', fn_val, '\n FP Validation:', fp_val, '\n TN Validation:', tn_val)
                train_metrics.append(tp_train)
                train_metrics.append(fn_train)
                train_metrics.append(fp_train)
                train_metrics.append(tn_train)
                val_metrics.append(tp_val)
                val_metrics.append(fn_val)
                val_metrics.append(fp_val)
                val_metrics.append(tn_val)

        if metric=='Mean Absolute Error (MAE)':
            from sklearn.metrics import mean_absolute_error as mae
            if tvs_flag==1:
                mae_train=mae(pred_train, y_train)
                mae_val=mae(pred_val, y_val)
                print('\n MAE for training set:', mae_train, '\n MAE for validation set:', mae_val)
                train_metrics.append(mae_train)
                val_metrics.append(mae_val)
            else:
                mae_train=mae(pred_train, y_train)
                mae_val=None
                print('\n MAE for training set:', mae_train, '\n MAE for validation set:', mae_val)
                train_metrics.append(mae_train)
                val_metrics.append(mae_val)

        if metric=='Mean Squared Error (MSE)':
            from sklearn.metrics import mean_squared_error as mse
            if tvs_flag==1:
                mse_train=mse(pred_train, y_train)
                mse_val=mse(pred_val, y_val)
                print('\n MSE for training set:', mse_train, '\n MSE for validation set:', mse_val)
                train_metrics.append(mse_train)
                val_metrics.append(mse_val)
            else:
                mse_train=mse(pred_train, y_train)
                mse_val=None
                print('\n MSE for training set:', mse_train, '\n MSE for validation set:', mse_val)
                train_metrics.append(mse_train)
                val_metrics.append(mse_val)

        if metric=='Root Mean Squared Error (RMSE)':
            from sklearn.metrics import mean_squared_error as rmse
            if tvs_flag==1:
                rmse_train=rmse(pred_train, y_train, squared=False)
                rmse_val=rmse(pred_val, y_val, squared=False)
                print('\n RMSE for training set:', rmse_train, '\n RMSE for validation set:', rmse_val)
                train_metrics.append(rmse_train)
                val_metrics.append(rmse_val)
            else:
                rmse_train=rmse(pred_train, y_train, squared=False)
                rmse_val=None
                print('\n RMSE for training set:', rmse_train, '\n RMSE for validation set:', rmse_val)
                train_metrics.append(rmse_train)
                val_metrics.append(rmse_val)

        if metric=='Root Mean Squared Log Error (RMSLE)':
            from sklearn.metrics import mean_squared_log_error as msle
            from math import sqrt
            sqrt=sqrt()
            if tvs_flag==1:
                rmsle_train=sqrt(msle(pred_train, y_train))
                rmsle_val=sqrt(msle(pred_val, y_val))
                print('\n RMSLE for training set:', rmsle_train, '\n RMSLE for validation set:', rmsle_val)
                train_metrics.append(rmsle_train)
                val_metrics.append(rmsle_val)
            else:
                rmsle_train=sqrt(msle(pred_train, y_train))
                rmsle_val=None
                print('\n RMSLE for training set:', rmsle_train, '\n RMSLE for validation set:', rmsle_val)
                train_metrics.append(rmsle_train)
                val_metrics.append(rmsle_val)

        if metric=='R2':
            from sklearn.metrics import r2_score as r2
            if tvs_flag==1:
                r2_train=r2(pred_train, y_train)
                r2_val=r2(pred_val, y_val)
                print('\n R2 score for training set:', r2_train, '\n R2 score for validation set:', r2_val)
                train_metrics.append(r2_train)
                val_metrics.append(r2_val)
            else:
                r2_train=r2(pred_train, y_train)
                r2_val=None
                print('\n R2 score for training set:', r2_train, '\n R2 score for validation set:', r2_val)
                train_metrics.append(r2_train)
                val_metrics.append(r2_val)

        if metric=='Adjusted R2':
            from sklearn.metrics import r2_score as r2
            if tvs_flag==1:
                r2_train=r2(pred_train, y_train)
                r2_val=r2(pred_val, y_val)
                ar2_train=(1-(1-r2_train))*((len(X_train)-1)/(len(X_train)-len(X_train.columns)-1))
                ar2_val=(1-(1-r2_val))*((len(X_val)-1)/(len(X_val)-len(X_val.columns)-1))
                print('\n Adjusted R2 score for training set:', ar2_train, '\n Adjusted R2 score for validation set:', ar2_val)
                train_metrics.append(ar2_train)
                val_metrics.append(ar2_val)
            else:
                r2_train=r2(pred_train, y_train)
                ar2_train=(1-(1-r2_train))*((len(X_train)-1)/(len(X_train)-len(X_train.columns)-1))
                ar2_val=None
                print('\n Adjusted R2 score for training set:', ar2_train, '\n Adjusted R2 score for validation set:', ar2_val)
                train_metrics.append(ar2_train)
                val_metrics.append(ar2_val)

    print('\n === Train metrics ===\n', train_metrics, '\n\n === Validation metrics ===\n', val_metrics)
    return (cmtdf, cmvdf, train_metrics, val_metrics)

def metric_table(model_id, evaluation_metrics, train_metrics, val_metrics):
    s = pd.DataFrame({'Training': train_metrics, 'Validation': val_metrics}, index=evaluation_metrics)
    print('\n\n ===== Metric DF =====\n', s)
    ddata=s.values.tolist()
    dcolumns=s.columns.tolist()
    dindex=s.index.tolist()
    for i, d in enumerate(ddata):
        d.insert(0, list(s.index)[i])
    dcolumns.insert(0, 'Model {}'.format(model_id))

    mdf = pd.DataFrame(columns=dcolumns, data=ddata)
    mdf.append(pd.Series(), ignore_index=True)
    print('\n\n ===== Metric final DF =====\n', mdf)
    return mdf

def publish_model(username, projectname, model, model_id, scale_flag):
    print('\n === Preparing Model to publish ===\n', model)
    try:
        os.makedirs(os.path.join(get_project_folder_path(username, projectname), "Models", "Published"))
    except:
        pass
    import pickle
    publish_name = 'Model_{}_{}.pkl'.format(model_id, scale_flag)
    #pkl_file = get_project_folder_path(username, projectname)+'\\'+publish_name
    pkl_file = os.path.join(get_project_folder_path(username, projectname), "Models", "Published", publish_name)
    data = model
    file = open(pkl_file,"wb")
    pickle.dump(data,file)
    file.close()
    print('\n === Actual model ===\n', model)
    print('\n === Model_{}_{} published ===\n'.format(model_id, scale_flag))
    return publish_name

def import_model(username, projectname, pub_name):
    #fp = get_project_folder_path(username, projectname)+'\\'+pub_name
    fp = os.path.join(get_project_folder_path(username, projectname), "Models", "Published", pub_name)
    print('FP:', fp)
    import pickle
    infile = open(fp,'rb')
    model = pickle.load(infile)
    infile.close()
    return model

def test_pred(username, projectname, model, scale_flag, X_test, model_id):
    if scale_flag != 0 : # Signifies scaled training set
        #scaler_path = get_project_folder_path(username, projectname) + '\\Training folder\\scaler_model_id_{}.pkl'.format(model_id)
        scaler_path = os.path.join(get_project_folder_path(username, projectname), "Training folder", "scaler_model_id_{}.pkl".format(model_id))
        import pickle
        infile = open(scaler_path,'rb')
        norm = pickle.load(infile)
        infile.close()
        X_test_norm = pd.DataFrame(norm.transform(X_test))
        X_test_norm.columns = X_test.columns
        X_test=X_test_norm.copy()

    pred_test=model.predict(X_test)
    return (pred_test)

def testing_pred_folder_creation(pred_test, X_test, test_folder_path):
    output_test=X_test.reset_index().copy()
    test_predictions=pd.DataFrame(pred_test)
    test_predictions.columns=['Predicted_target']
    #val_target.columns=['Actual_target']
    output_test=pd.concat([output_test, test_predictions], axis=1)
    #output_test.to_csv(test_folder_path+'\\Predictions_for_TestSet.csv', index=False)
    output_test.to_csv(os.path.join(test_folder_path, "Predictions_for_TestSet.csv"), index=False)
    print('\n === Predictions for test set created ===\n')

def get_line_box_contour_swarm_bar_plots(df, pt, x, y, hue):
    if hue == 'no':
        hue = None
    if pt == 'line':
        if x =='Over index':
            try:
                fig = px.line(data_frame=df, x=df.index, y=y, color=hue)
            except:
                return 1
            fig.update_layout(title='Line plot: "{}" across dataset'.format(y))
            return (fig)

        else:
            try:
                fig = px.line(data_frame=df, x=x, y=y, color=hue)
            except:
                return 1
            fig.update_layout(title='Line plot: "{}" vs "{}"'.format(y, x))
            return fig
    if pt == 'box':
        try:
            fig=px.box(df, x=x, y=y, color=hue)
        except:
            return 1
        fig.update_layout(title='Box plot of {} vs {} columns'.format(y, x))
        return (fig)
    if pt == 'contour':
        try:
            fig=px.density_contour(df, x=x, y=y, color=hue, marginal_x="histogram", marginal_y="histogram")
        except:
            return 1
        fig.update_layout(title='Density Contour plot of {} vs {} columns'.format(y, x))
        return (fig)
    if pt == 'swarm':
        try:
            fig=px.strip(df, x=x, y=y, color=hue)
        except:
            return 1
        fig.update_layout(title='Swarm plot of {} vs {} columns'.format(y, x))
        return (fig)
    if pt == 'bar':
        try:
            fig=px.bar(data_frame=df, x=x, y=y, color=hue)
        except:
            return 1
        fig.update_layout(title='Bar plot for {} vs {} columns'.format(y, x))
        return (fig)

def get_scatter_plots(df, x, y, hue, skr):
    if hue == 'no':
        hue = None
    if skr == 'yes': # Keep Reg Line
        try:
            fig = px.scatter(data_frame=df, x=x, y=y, trendline='ols', color=hue)
        except:
            return 1
        fig.update_layout(title='Scatter plot of "{}" vs "{}" columns'.format(y, x))
        return (fig)
    else: # No Reg Line
        try:
            fig = px.scatter(data_frame=df, x=x, y=y, color=hue)
        except:
            return 1
        fig.update_layout(title='Scatter plot of "{}" vs "{}" columns'.format(y, x))
        return (fig)

def get_heatmap_plots(df, type):
    if type != 'correlation':
        try:
            fig=px.imshow(df)
        except:
            return 1
        fig.update_layout(title='Heatmap for complete Dataset')
        return (fig)
    else:
        try:
            fig=px.imshow(df.corr())
        except:
            return 1
        fig.update_layout(title='Correlation Heatmap')
        return (fig)

def get_histogram_plots(df, x, hue, bins):
    if hue == 'no':
        hue = None
    if bins == 'no':
        bins = None
    elif bins != 'no':
        try:
            bins = int(bins)
        except:
            return 1
    try:
        fig=px.histogram(df, x=x, color=hue, nbins=bins, barmode='group') # nbins=10
    except:
        return 1
    fig.update_layout(title='Histogram of {} column'.format(x))
    return (fig)

def get_replace_num_cols(sel_num_col, rtype, ov, nv, df):
    if rtype == 'gt': # greater than
        df[sel_num_col].values[df[sel_num_col].values>int(ov)]=int(nv)
        return df
    if rtype == 'lt': # lesser than
        df[sel_num_col].values[df[sel_num_col].values<int(ov)]=int(nv)
        return df
    if rtype == 'uv': # unique value
        df[sel_num_col]=df[sel_num_col].replace([int(ov)], int(nv))
        return df

def get_replace_cat_cols(sel_cat_col, ov, nv, df):
    df[sel_cat_col]=df[sel_cat_col].replace([ov], nv)
    return df

def get_fill_num_cols(df, sel_num_col, ftype, uv):
    num_cols=[col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    if ftype == 'mean':
        if sel_num_col=='All numerical columns':
            df[num_cols]=df[num_cols].fillna(value=df[num_cols].mean())
        else:
            df[sel_num_col]=df[sel_num_col].fillna(value=df[sel_num_col].mean())
        return df
    if ftype == 'median':
        if sel_num_col=='All numerical columns':
            df[num_cols]=df[num_cols].fillna(value=df[num_cols].median())
        else:
            df[sel_num_col]=df[sel_num_col].fillna(value=df[sel_num_col].median())
        return df
    if ftype == 'mode':
        if sel_num_col=='All numerical columns':
            df[num_cols]=df[num_cols].fillna(value=df[num_cols].mode().iloc[0], axis=0)
        else:
            df[sel_num_col]=df[sel_num_col].fillna(value=df[sel_num_col].mode().iloc[0], axis=0)
        return df
    if ftype == 'interpolation':
        if sel_num_col=='All numerical columns':
            df[num_cols]=df[num_cols].interpolate(method ='linear', limit_direction ='both')
        else:
            df[sel_num_col]=df[sel_num_col].interpolate(method ='linear', limit_direction ='both')
        return df
    if ftype == 'previous':
        if sel_num_col=='All numerical columns':
            df[num_cols]=df[num_cols].interpolate(method ='pad')
        else:
            df[sel_num_col]=df[sel_num_col].interpolate(method ='pad')
        return df
    if ftype == 'unique':
        if sel_num_col=='All numerical columns':
            df[num_cols]=df[num_cols].fillna(value=int(uv))
        else:
            df[sel_num_col]=df[sel_num_col].fillna(value=int(uv))
        return df

def get_fill_cat_cols(df, sel_cat_col, ftype, uv):
    cat_cols=[col for col in df.columns if df[col].dtype == 'object']
    if ftype == 'most_frequent':
        if sel_cat_col=='All categorical columns':
            df[cat_cols]=df[cat_cols].fillna(value=df[cat_cols].mode().iloc[0], axis=0)
        else:
            df[sel_cat_col]=df[sel_cat_col].fillna(value=df[sel_cat_col].mode().iloc[0], axis=0)
        return df
    if ftype == 'unique':
        if sel_cat_col=='All categorical columns':
            df[cat_cols]=df[cat_cols].fillna(value=uv)
        else:
            df[sel_cat_col]=df[sel_cat_col].fillna(value=uv)
        return df

def get_convert_cols(df, col, dtype):
        try:
            df[col]=df[col].astype(dtype)
            return (df, None)
        except:
            return (df, 1)

def get_encode_cols(df, col):
    if col=='All categorical columns':
        cat_cols=[col for col in df.columns if df[col].dtype == 'object']
        try:
            for acols in cat_cols:
                df[acols]=df[acols].astype('category').cat.codes.astype('int64')
        except:
            return (df, 1)
    else:
        df[col]=df[col].astype('category').cat.codes.astype('int64')
    return (df, None)

# ========== END (Non-API functions & classes) ==========



# ========== API START ==========
app = Flask(__name__)

# ===== API Definitions START =====
@app.route('/')
def hello_world():
    return 'HELLO WORLD\nThis is a Test message'

@app.route('/user/login', methods=['GET', 'POST']) # TBA
def user_login():
    if request.method=='POST':
        user = request.get_json()
        # Authenticate the username & password officially
        # Assuming that the username & password are correct for now
        un = user['username']
        """
        try:
            os.mkdir(client_folder_path)
        except:
            pass
        os.chdir(client_folder_path)
        #print(get_project_folder_path(username, projectname))
        #user_folder_path = client_folder_path+'\\{}'.format(un)
        user_folder_path = os.path.join(client_folder_path, un)
        try:
            os.mkdir(user_folder_path)
        except:
            pass
        os.chdir(user_folder_path)
        #print(get_project_folder_path(username, projectname))
        #projects_folder_path = user_folder_path+'\\Projects'
        projects_folder_path = os.path.join(user_folder_path, "Projects")
        try:
            os.mkdir(projects_folder_path)
        except:
            pass
        os.chdir(projects_folder_path)
        """
        return jsonify({'username':user['username'], 'login_status':'Successful'})

@app.route('/menu/new', methods=['GET', 'POST'])
def menu_new():
    if request.method=='GET':
        #user = request.get_json()
        #un = user['username']
        un = request.args.get('username')
        #os.chdir(client_folder_path+'\\{}'.format(un)+'\\Projects')
        #os.chdir(os.path.join(client_folder_path, un, "Projects"))
        # LOAD JSON
        #with open('C:\\Users\\DELL\\Desktop\\Toolkit\\jct.json') as file:
        #with open(user_json_path+'\\'+user_json_filename) as file:
        # LOAD JCT JSON
        with open(os.path.join(get_user_json_file_path())) as file:
            main_format=json.load(file)

        for ud in main_format:
            if ud["username"]==un:
                ud["project_id_counter"] = ud["project_id_counter"] + 1
                print('\nNew project ID is:', ud["project_id_counter"])
                return jsonify({'username':un, 'new_project_id':ud["project_id_counter"]})

@app.route('/menu/new/confirm', methods=['GET', 'POST'])
def new_ok():
    if request.method=='POST':
        pds_dict = request.get_json()
        username = pds_dict['username']
        project_id = pds_dict['project_id']
        projectname = 'Project {}'.format(project_id)
        #print(pds_dict)
        # LOAD JSON
        #with open(user_json_path+'\\'+user_json_filename) as file:
        with open(os.path.join(get_user_json_file_path())) as file:
            main_format=json.load(file)
        for ud in main_format:
            if ud["username"]==pds_dict['username']:
                ud["project_id_counter"] = int(pds_dict['project_id'])

                #os.mkdir('Project {}'.format(ud["project_id_counter"]))
                #os.chdir('Project {}'.format(ud["project_id_counter"]))

                metadata_main_format = get_metadata_main_format()
                metadata_main_format['project_description']=pds_dict['project_description']

                #DUMP Metadata JSON /// First time
                #with open(get_project_folder_path(username, projectname)+'\\Metadata.json', 'w') as file:
                write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))


                ud['project_details'].append({
                "project_id": ud["project_id_counter"],
                "project_description": pds_dict['project_description']})
                #DUMP JSON
                #with open(user_json_path+'\\'+user_json_filename, 'w') as file:
                with open(os.path.join(get_user_json_file_path()), 'w') as file:
                    json.dump(main_format, file, indent=4)
                #print('DONE')
                return jsonify({"project_id": ud["project_id_counter"], "project_description": pds_dict['project_description']})

@app.route('/menu/open', methods=['GET', 'POST'])
def menu_open():
    if request.method=='GET':

        # LOAD JSON
        #with open(user_json_path+'\\'+user_json_filename) as file:
        with open(os.path.join(get_user_json_file_path())) as file:
            main_format=json.load(file)
        #user = request.get_json()
        #un = user['username']
        un = request.args.get('username')
        #os.chdir(client_folder_path+'\\{}'.format(un)+'\\Projects')
        project_list = []
        for ud in main_format:
            if ud["username"]==un:
                if len(ud['project_details'])==0:
                    return jsonify({"status": "No projects available", "action": "Create your first project by clicking 'New'"})
                for up in ud['project_details']:
                    project_list.append('Project {}'.format(up['project_id']))
        #print('\nProject list is: ', project_list)
        return jsonify(project_list)

@app.route('/menu/open/project_selection', methods=['GET', 'POST'])
def open_project_sel():
    if request.method=='GET':
        # LOAD JSON
        #with open(user_json_path+'\\'+user_json_filename) as file:
        with open(os.path.join(get_user_json_file_path())) as file:
            main_format=json.load(file)
        #user = request.get_json()
        #un = user['username']
        #pid = user['project_id_selected']
        un = request.args.get('username')
        pid = request.args.get('project_id_selected')
        #print('\n\n\nun:{}, pid:{}, {}\n\n\n'.format(un, pid, type(pid)))
        for ud in main_format:
            #print(ud)
            if ud["username"]==un:
                for up in ud['project_details']:
                    #print(up)
                    if up['project_id']==int(pid):
                        print('\n\n\n',up)
                        return jsonify(up)

@app.route('/menu/open/confirm', methods=['GET', 'POST'])
def open_ok():
    #global train_df
    #global train_flag
    #global test_df
    #global test_flag
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #os.chdir(client_folder_path+'\\{}'.format(username)+'\\Projects\\Project {}'.format(project_id))
    #os.chdir(os.path.join(client_folder_path, username, "Projects", "Project {}".format(project_id)))
    #print('\n === CWD ===\n', get_project_folder_path(username, projectname))

    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    print('\n === METADATA ===\n', metadata_main_format)

    if metadata_main_format['input']['training_set'] != 0:
        #train_df = pd.read_csv(get_project_folder_path(username, projectname)+'\\Training folder\\training_dataset_input.csv')
        train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "training_dataset_input.csv"))
        train_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"), index=False)
        train_csv_return_format = read_csv_file(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))
        train_flag = 1
    else:
        train_df = 'Unavailable'
        train_flag = 0

    if metadata_main_format['input']['testing_set'] != 0:
        #test_df = pd.read_csv(get_project_folder_path(username, projectname)+'\\Testing folder\\testing_dataset_input.csv')
        test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "testing_dataset_input.csv"))
        test_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"), index=False)
        test_csv_return_format = read_csv_file(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))
        test_flag = 1
    else:
        test_df = 'Unavailable'
        test_flag = 0

    '''
    for ops in ['drop', 'replace', 'fill', 'convert', 'encode']:
        if metadata_main_format['clean_train_set'][ops]==1:
            #train_df = pd.read_csv(get_project_folder_path(username, projectname)+'\\Training folder\\current_training_dataset_after_cleaning.csv')
            train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))
            train_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"), index=False)
            train_flag = 1
            break
    for ops in ['drop', 'replace', 'fill', 'convert', 'encode']:
        if metadata_main_format['clean_test_set'][ops]==1:
            #test_df = pd.read_csv(get_project_folder_path(username, projectname)+'\\Testing folder\\current_testing_dataset_after_cleaning.csv')
            test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))
            test_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"), index=False)
            test_flag = 1
            break
    '''

    return jsonify(training_set=train_csv_return_format, testing_set=test_csv_return_format, metadata=metadata_main_format)




@app.route('/workflow/training/data/browse', methods=['GET', 'POST'])
def browse_training():
    # some code
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)
    #global train_df
    #global train_flag
    if request.method=='POST':
        input_file = request.files['file']
        print('\n\n\n =====',type(input_file))
        try:
            os.mkdir(os.path.join(get_project_folder_path(username, projectname), "Training folder"))
        except:
            pass
        #input_file.save(get_project_folder_path(username, projectname)+'\\Training folder\\training_dataset_input.csv')
        input_file.save(os.path.join(get_project_folder_path(username, projectname), "Training folder", "training_dataset_input.csv"))
        # LOAD Metadata JSON
        metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
        metadata_main_format['input']['training_set']=input_file.filename
        metadata_main_format['input']['train_flag']=1
        #DUMP Metadata JSON
        write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

        train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "training_dataset_input.csv"))
        train_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"), index=False)
        csv_return_format = read_csv_file(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))
        #train_flag = 1
        print('\n\n\n ===== Browse train =====\n\n\n', train_df.head())
        #train_df_dict = train_df.to_dict(orient='split')
        #train_df_dict['shape']=train_df.shape
        #train_df_dict['file_name']=input_file.filename
        #return jsonify(train_df_dict)
        #crf = json.dumps(csv_return_format, cls=DecimalEncoder)
        #print('\n === BROWSE CRF TYPE===\n', type(crf))
        return jsonify(Training_dataset=csv_return_format, current_shape_of_training_data=train_df.shape)

@app.route('/workflow/testing/data/browse', methods=['GET', 'POST'])
def browse_testing():
    # some code
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)
    #global test_df
    #global test_flag
    if request.method=='POST':
        input_file = request.files['file']
        print('\n\n\n =====',type(input_file))
        try:
            os.mkdir(os.path.join(get_project_folder_path(username, projectname), "Testing folder"))
        except:
            pass
        #input_file.save(get_project_folder_path(username, projectname)+'\\Testing folder\\testing_dataset_input.csv')
        input_file.save(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "testing_dataset_input.csv"))
        # LOAD Metadata JSON
        metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
        metadata_main_format['input']['testing_set']=input_file.filename
        metadata_main_format['input']['test_flag']=1
        #DUMP Metadata JSON
        write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

        test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "testing_dataset_input.csv"))
        test_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"), index=False)
        csv_return_format = read_csv_file(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))
        #test_flag = 1
        print('\n\n\n ===== Browse test =====\n\n\n', test_df.head())
        #test_df_dict = test_df.to_dict(orient='split')
        #test_df_dict['shape']=test_df.shape
        #test_df_dict['file_name']=input_file.filename
        #return jsonify(test_df_dict)
        #crf = json.dumps(csv_return_format, cls=DecimalEncoder)
        return jsonify(Testing_dataset=csv_return_format, current_shape_of_testing_data=test_df.shape)




@app.route('/workflow/training/data/data', methods=['GET', 'POST'])
def data_show_train():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)
    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))

    #global train_flag
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    train_flag = metadata_main_format['input']['train_flag']

    if train_flag != 1:
        return jsonify({"status": "No dataset selected", "action": "Kindly browse the dataset first"})

    #train_df_dict = train_df.to_dict(orient='split')
    #train_df_dict['shape']=train_df.shape
    #return jsonify(train_df_dict)
    try:
        os.mkdir(os.path.join(get_project_folder_path(username, projectname), "CSVsummary"))
    except:
        pass
    #train_df.to_csv(get_project_folder_path(username, projectname)+'\\CSVsummary\\current_data_train.csv', index=False)
    train_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_data_train.csv"), index=False)
    csv_return_format = read_csv_file(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_data_train.csv"))
    #crf = json.dumps(csv_return_format, cls=DecimalEncoder)
    return jsonify(Training_data_show=csv_return_format, current_shape_of_training_data=train_df.shape)

@app.route('/workflow/testing/data/data', methods=['GET', 'POST'])
def data_show_test():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)
    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))

    #global test_flag
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    test_flag = metadata_main_format['input']['test_flag']

    if test_flag != 1:
        return jsonify({"status": "No dataset selected", "action": "Kindly browse the dataset first"})

    #test_df_dict = test_df.to_dict(orient='split')
    #test_df_dict['shape']=test_df.shape
    #return jsonify(test_df_dict)
    try:
        os.mkdir(os.path.join(get_project_folder_path(username, projectname), "CSVsummary"))
    except:
        pass
    test_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_data_test.csv"), index=False)
    csv_return_format = read_csv_file(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_data_test.csv"))
    #crf = json.dumps(csv_return_format, cls=DecimalEncoder)
    return jsonify(Testing_data_show=csv_return_format, current_shape_of_testing_data=test_df.shape)

@app.route('/workflow/training/data/head', methods=['GET', 'POST'])
def head_show_train():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)
    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))

    #global train_flag
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    train_flag = metadata_main_format['input']['train_flag']

    if train_flag != 1:
        return jsonify({"status": "No dataset selected", "action": "Kindly browse the dataset first"})
    train_df_head = train_df.head()
    train_df_head_dict = train_df_head.to_dict(orient='split')
    return jsonify(train_df_head_dict)

@app.route('/workflow/testing/data/head', methods=['GET', 'POST'])
def head_show_test():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)
    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))

    #global test_flag
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    test_flag = metadata_main_format['input']['test_flag']

    if test_flag != 1:
        return jsonify({"status": "No dataset selected", "action": "Kindly browse the dataset first"})
    test_df_head = test_df.head()
    test_df_head_dict = test_df_head.to_dict(orient='split')
    return jsonify(test_df_head_dict)

@app.route('/workflow/training/data/tail', methods=['GET', 'POST'])
def tail_show_train():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)
    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))

    #global train_flag
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    train_flag = metadata_main_format['input']['train_flag']

    if train_flag != 1:
        return jsonify({"status": "No dataset selected", "action": "Kindly browse the dataset first"})

    train_df_tail = train_df.tail()
    train_df_tail_dict = train_df_tail.to_dict(orient='split')
    return jsonify(train_df_tail_dict)

@app.route('/workflow/testing/data/tail', methods=['GET', 'POST'])
def tail_show_test():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)
    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))

    #global test_flag
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    test_flag = metadata_main_format['input']['test_flag']

    if test_flag != 1:
        return jsonify({"status": "No dataset selected", "action": "Kindly browse the dataset first"})

    test_df_tail = test_df.tail()
    test_df_tail_dict = test_df_tail.to_dict(orient='split')
    return jsonify(test_df_tail_dict)

@app.route('/workflow/training/data/describe', methods=['GET', 'POST'])
def describe_show_train():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))

    #global train_flag
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    train_flag = metadata_main_format['input']['train_flag']

    if train_flag != 1:
        return jsonify({"status": "No dataset selected", "action": "Kindly browse the dataset first"})

    des=train_df.describe()
    ddata=des.values.tolist()
    dcolumns=des.columns.tolist()
    dindex=des.index.tolist()
    #for i, d in enumerate(ddata):
    #    d.insert(0, list(des.index)[i])
    #dcolumns.insert(0, 'Stat')
    train_df_describe_dict={"columns":dcolumns, "data":ddata, "index":dindex}
    ddf=pd.DataFrame(columns=dcolumns, data=ddata, index=dindex)
    try:
        os.mkdir(os.path.join(get_project_folder_path(username, projectname), "CSVsummary"))
    except:
        pass
    #ddf.to_csv(get_project_folder_path(username, projectname)+'\\CSVsummary\\current_description_train.csv')
    ddf.to_csv(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_description_train.csv"), index=False)
    csv_return_format = read_csv_file(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_description_train.csv"))
    #return jsonify(train_df_describe_dict)
    #crf = json.dumps(csv_return_format, cls=DecimalEncoder)
    return jsonify(training_data_statistical_description=csv_return_format)

@app.route('/workflow/testing/data/describe', methods=['GET', 'POST'])
def describe_show_test():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))

    #global test_flag
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    test_flag = metadata_main_format['input']['test_flag']

    if test_flag != 1:
        return jsonify({"status": "No dataset selected", "action": "Kindly browse the dataset first"})

    des=test_df.describe()
    ddata=des.values.tolist()
    dcolumns=des.columns.tolist()
    dindex=des.index.tolist()
    #for i, d in enumerate(ddata):
    #    d.insert(0, list(des.index)[i])
    #dcolumns.insert(0, 'Stat')
    test_df_describe_dict={"columns":dcolumns, "data":ddata, "index":dindex}
    ddf=pd.DataFrame(columns=dcolumns, data=ddata, index=dindex)
    try:
        os.mkdir(os.path.join(get_project_folder_path(username, projectname), "CSVsummary"))
    except:
        pass
    #ddf.to_csv(get_project_folder_path(username, projectname)+'\\CSVsummary\\current_description_test.csv')
    ddf.to_csv(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_description_test.csv"), index=False)
    csv_return_format = read_csv_file(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_description_test.csv"))
    #return jsonify(test_df_describe_dict)
    #crf = json.dumps(csv_return_format, cls=DecimalEncoder)
    #print('\n === DESCRIBE CRF TYPE===\n', type(crf))
    return jsonify(testing_data_statistical_description=csv_return_format)

@app.route('/workflow/training/data/info', methods=['GET', 'POST'])
def info_show_train():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))

    #global train_flag
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    train_flag = metadata_main_format['input']['train_flag']

    if train_flag != 1:
        return jsonify({"status": "No dataset selected", "action": "Kindly browse the dataset first"})

    s={'Columns': train_df.columns,
       'dtypes':train_df.dtypes.astype(str).values.tolist(),
       'Null values': train_df.isnull().sum().values.tolist(),
       '% Null values': round((train_df.isnull().sum()/len(train_df)*100),2).values.tolist()}
    dtdf=pd.DataFrame(s)
    #idata=dtdf.values.tolist()
    #icolumns=dtdf.columns.tolist()
    #iindex=dtdf.index.tolist()
    #train_df_info_dict={"columns": icolumns, "data": idata, "index": iindex}
    #print('\n\n\n')
    #print(train_df_info_dict)
    #idf=pd.DataFrame(train_df_info_dict)
    try:
        os.mkdir(os.path.join(get_project_folder_path(username, projectname), "CSVsummary"))
    except:
        pass
    #dtdf.to_csv(get_project_folder_path(username, projectname)+'\\CSVsummary\\current_info_train.csv', index=False)
    dtdf.to_csv(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_info_train.csv"), index=False)
    csv_return_format = read_csv_file(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_info_train.csv"))
    #return jsonify(train_df_info_dict)
    #crf = json.dumps(csv_return_format, cls=DecimalEncoder)
    return jsonify(training_data_info=csv_return_format)

@app.route('/workflow/testing/data/info', methods=['GET', 'POST'])
def info_show_test():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))

    #global test_flag
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    test_flag = metadata_main_format['input']['test_flag']

    if test_flag != 1:
        return jsonify({"status": "No dataset selected", "action": "Kindly browse the dataset first"})

    s={'Columns': test_df.columns,
       'dtypes':test_df.dtypes.astype(str).values.tolist(),
       'Null values': test_df.isnull().sum().values.tolist(),
       '% Null values': round((test_df.isnull().sum()/len(test_df)*100),2).values.tolist()}
    dtdf=pd.DataFrame(s)
    #idata=dtdf.values.tolist()
    #icolumns=dtdf.columns.tolist()
    #iindex=dtdf.index.tolist()
    #test_df_info_dict={"columns": icolumns, "data": idata, "index": iindex}
    try:
        os.mkdir(os.path.join(get_project_folder_path(username, projectname), "CSVsummary"))
    except:
        pass
    #dtdf.to_csv(get_project_folder_path(username, projectname)+'\\CSVsummary\\current_info_test.csv', index=False)
    dtdf.to_csv(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_info_test.csv"), index=False)
    csv_return_format = read_csv_file(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_info_test.csv"))
    #return jsonify(test_df_info_dict)
    #crf = json.dumps(csv_return_format, cls=DecimalEncoder)
    return jsonify(testing_data_info=csv_return_format)




@app.route('/workflow/training/column_combo', methods=['GET', 'POST'])
def column_combo_train():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))
    #global train_flag
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    train_flag = metadata_main_format['input']['train_flag']

    if train_flag != 1:
        return jsonify({"status": "No dataset selected", "action": "Kindly browse the dataset first"})
    columns = train_df.columns.tolist()
    return jsonify(current_available_columns_of_train_set=columns)

@app.route('/workflow/testing/column_combo', methods=['GET', 'POST'])
def column_combo_test():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))
    #global test_flag
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    test_flag = metadata_main_format['input']['test_flag']

    if test_flag != 1:
        return jsonify({"status": "No dataset selected", "action": "Kindly browse the dataset first"})
    columns = test_df.columns.tolist()
    return jsonify(current_available_columns_of_test_set=columns)

@app.route('/workflow/training_testing/dtype_combo', methods=['GET', 'POST'])
def dtpye_combo_train_test():
    dtype_list=['int64', 'float64', 'object']
    return jsonify(data_types_available=dtype_list)

@app.route('/workflow/training/num_cols_combo', methods=['GET', 'POST'])
def num_cols_combo_train():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))
    #global train_flag
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    train_flag = metadata_main_format['input']['train_flag']

    if train_flag != 1:
        return jsonify({"status": "No dataset selected", "action": "Kindly browse the dataset first"})
    num_cols=[col for col in train_df.columns if train_df[col].dtype in ['int64', 'float64']]
    return jsonify(numerical_columns_of_train_set=num_cols)

@app.route('/workflow/testing/num_cols_combo', methods=['GET', 'POST'])
def num_cols_combo_test():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))
    #global test_flag
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    test_flag = metadata_main_format['input']['test_flag']

    if test_flag != 1:
        return jsonify({"status": "No dataset selected", "action": "Kindly browse the dataset first"})
    num_cols=[col for col in test_df.columns if test_df[col].dtype in ['int64', 'float64']]
    return jsonify(numerical_columns_of_test_set=num_cols)

@app.route('/workflow/training/cat_cols_combo', methods=['GET', 'POST'])
def cat_cols_combo_train():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))
    #global train_flag
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    train_flag = metadata_main_format['input']['train_flag']

    if train_flag != 1:
        return jsonify({"status": "No dataset selected", "action": "Kindly browse the dataset first"})
    cat_cols=[col for col in train_df.columns if train_df[col].dtype == 'object']
    return jsonify(categorical_columns_of_train_set=cat_cols)

@app.route('/workflow/testing/cat_cols_combo', methods=['GET', 'POST'])
def cat_cols_combo_test():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))
    #global test_flag
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    test_flag = metadata_main_format['input']['test_flag']

    if test_flag != 1:
        return jsonify({"status": "No dataset selected", "action": "Kindly browse the dataset first"})
    cat_cols=[col for col in test_df.columns if test_df[col].dtype == 'object']
    return jsonify(categorical_columns_of_test_set=cat_cols)

@app.route('/workflow/training/model/algorithm_combo')
def algorithm_combo():
    algorithms=[
    'Logistic Regression',
    'KNN Classifier',
    'SVM',
    'Decision Tree Classifier',
    'Random Forest Classifier',
    'Linear Regression',
    'KNN Regressor',
    'SVR',
    'Decision Tree Regressor',
    'Random Forest Regressor'
    ]
    return jsonify(available_algorithms=algorithms)




@app.route('/workflow/training/clean/drop/drop_1', methods=['GET', 'POST'])
def train_drop_rows():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))
    indexes_list = request.get_json() # getting list of index/indexes to be dropped
    train_df = train_df.drop(index = indexes_list, axis=0).reset_index(drop=True)
    #train_df.to_csv(get_project_folder_path(username, projectname)+'\\Training folder\\current_training_dataset_after_cleaning.csv', index=False)
    train_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"), index=False)
    csv_return_format=read_csv_file(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    metadata_main_format['clean_train_set']['drop'] = 1
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    return jsonify(modified_train_dataset=csv_return_format, shape=train_df.shape)

@app.route('/workflow/testing/clean/drop/drop_1', methods=['GET', 'POST'])
def test_drop_rows():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))
    indexes_list = request.get_json() # getting list of index/indexes to be dropped
    test_df = test_df.drop(index = indexes_list, axis=0).reset_index(drop=True)
    #test_df.to_csv(get_project_folder_path(username, projectname)+'\\Testing folder\\current_testing_dataset_after_cleaning.csv', index=False)
    test_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"), index=False)
    csv_return_format=read_csv_file(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    metadata_main_format['clean_test_set']['drop'] = 1
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    return jsonify(modified_test_dataset=csv_return_format, shape=test_df.shape)

@app.route('/workflow/training/clean/drop/drop_2', methods=['GET', 'POST'])
def train_drop_columns():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))
    columns_list = request.get_json() # list of column/columns to be dropped
    train_df = train_df.drop(columns=columns_list, axis=1)
    train_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"), index=False)
    csv_return_format=read_csv_file(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    metadata_main_format['clean_train_set']['drop'] = 1
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    return jsonify(modified_train_dataset=csv_return_format, shape=train_df.shape)

@app.route('/workflow/testing/clean/drop/drop_2', methods=['GET', 'POST'])
def test_drop_columns():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))
    columns_list = request.get_json() # getting list of column/columns to be dropped
    test_df = test_df.drop(columns = columns_list, axis=1)
    #test_df.to_csv(get_project_folder_path(username, projectname)+'\\Testing folder\\current_testing_dataset_after_cleaning.csv', index=False)
    test_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"), index=False)
    csv_return_format=read_csv_file(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    metadata_main_format['clean_test_set']['drop'] = 1
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    return jsonify(modified_test_dataset=csv_return_format, shape=test_df.shape)

@app.route('/workflow/training/clean/replace/replace_1', methods=['GET', 'POST'])
def train_replace_num_cols():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))
    rep_json = request.get_json()
    sel_num_col = rep_json['col']
    rtype = rep_json['replace_type']
    ov = rep_json['old_value']
    nv = rep_json['new_value']

    train_df = get_replace_num_cols(sel_num_col, rtype, ov, nv, train_df)
    train_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"), index=False)
    csv_return_format=read_csv_file(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    metadata_main_format['clean_train_set']['replace'] = 1
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    return jsonify(modified_train_dataset=csv_return_format, shape=train_df.shape)

@app.route('/workflow/testing/clean/replace/replace_1', methods=['GET', 'POST'])
def test_replace_num_cols():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))
    rep_json = request.get_json()
    sel_num_col = rep_json['col']
    rtype = rep_json['replace_type']
    ov = rep_json['old_value']
    nv = rep_json['new_value']

    test_df = get_replace_num_cols(sel_num_col, rtype, ov, nv, test_df)
    test_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"), index=False)
    csv_return_format=read_csv_file(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    metadata_main_format['clean_test_set']['replace'] = 1
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    return jsonify(modified_test_dataset=csv_return_format, shape=test_df.shape)

@app.route('/workflow/training/clean/replace/replace_2', methods=['GET', 'POST'])
def train_replace_cat_cols():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))
    rep_json = request.get_json()
    sel_cat_col = rep_json['col']
    ov = rep_json['old_value']
    nv = rep_json['new_value']

    train_df = get_replace_cat_cols(sel_cat_col, ov, nv, train_df)
    train_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"), index=False)
    csv_return_format=read_csv_file(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    metadata_main_format['clean_train_set']['replace'] = 1
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    return jsonify(modified_train_dataset=csv_return_format, shape=train_df.shape)

@app.route('/workflow/testing/clean/replace/replace_2', methods=['GET', 'POST'])
def test_replace_cat_cols():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))
    rep_json = request.get_json()
    sel_cat_col = rep_json['col']
    ov = rep_json['old_value']
    nv = rep_json['new_value']

    test_df = get_replace_cat_cols(sel_cat_col, ov, nv, test_df)
    test_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"), index=False)
    csv_return_format=read_csv_file(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    metadata_main_format['clean_test_set']['replace'] = 1
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    return jsonify(modified_train_dataset=csv_return_format, shape=test_df.shape)

@app.route('/workflow/training/clean/fill/fill_1', methods=['GET', 'POST'])
def train_fill_num_cols():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))

    fill_json = request.get_json()
    sel_num_col = fill_json['col']
    ftype = fill_json['fill_type']
    uv = fill_json['new_value']

    train_df = get_fill_num_cols(train_df, sel_num_col, ftype, uv)
    train_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"), index=False)
    csv_return_format=read_csv_file(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    metadata_main_format['clean_train_set']['fill'] = 1
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    return jsonify(modified_train_dataset=csv_return_format, shape=train_df.shape)

@app.route('/workflow/testing/clean/fill/fill_1', methods=['GET', 'POST'])
def test_fill_num_cols():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))

    fill_json = request.get_json()
    sel_num_col = fill_json['col']
    ftype = fill_json['fill_type']
    uv = fill_json['new_value']

    test_df = get_fill_num_cols(test_df, sel_num_col, ftype, uv)
    test_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"), index=False)
    csv_return_format=read_csv_file(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    metadata_main_format['clean_test_set']['fill'] = 1
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    return jsonify(modified_test_dataset=csv_return_format, shape=test_df.shape)

@app.route('/workflow/training/clean/fill/fill_2', methods=['GET', 'POST'])
def train_fill_cat_cols():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))

    fill_json = request.get_json()
    sel_cat_col = fill_json['col']
    ftype = fill_json['fill_type']
    uv = fill_json['new_value']

    train_df = get_fill_cat_cols(train_df, sel_cat_col, ftype, uv)
    train_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"), index=False)
    csv_return_format=read_csv_file(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    metadata_main_format['clean_train_set']['fill'] = 1
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    return jsonify(modified_train_dataset=csv_return_format, shape=train_df.shape)

@app.route('/workflow/testing/clean/fill/fill_2', methods=['GET', 'POST'])
def test_fill_cat_cols():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))

    fill_json = request.get_json()
    sel_cat_col = fill_json['col']
    ftype = fill_json['fill_type']
    uv = fill_json['new_value']

    test_df = get_fill_cat_cols(test_df, sel_cat_col, ftype, uv)
    test_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"), index=False)
    csv_return_format=read_csv_file(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    metadata_main_format['clean_test_set']['fill'] = 1
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    return jsonify(modified_test_dataset=csv_return_format, shape=test_df.shape)

@app.route('/workflow/training/clean/convert/convert', methods=['GET', 'POST'])
def train_convert_col():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))

    convert_json = request.get_json()
    col = convert_json['col']
    dtype = convert_json['dtype']

    train_df, flag = get_convert_cols(train_df, col, dtype)
    try:
        if Flag == 1:
            return 'Could not convert. Remove all NaN value first'
    except:
        pass
    train_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"), index=False)
    csv_return_format=read_csv_file(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    metadata_main_format['clean_train_set']['convert'] = 1
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    return jsonify(modified_train_dataset=csv_return_format, shape=train_df.shape)

@app.route('/workflow/testing/clean/convert/convert', methods=['GET', 'POST'])
def test_convert_col():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))

    convert_json = request.get_json()
    col = convert_json['col']
    dtype = convert_json['dtype']

    test_df, flag = get_convert_cols(test_df, col, dtype)
    try:
        if flag == 1:
            return 'Could not convert. Remove all NaN value first'
    except:
        pass
    test_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"), index=False)
    csv_return_format=read_csv_file(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    metadata_main_format['clean_test_set']['convert'] = 1
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    return jsonify(modified_test_dataset=csv_return_format, shape=test_df.shape)

@app.route('/workflow/training/clean/encode/encode', methods=['GET', 'POST'])
def train_encode_col():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))

    encode_json = request.get_json()
    col = encode_json['col']

    train_df, flag = get_encode_cols(train_df, col)
    try:
        if flag == 1:
            return 'Could not encode'
    except:
        pass
    train_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"), index=False)
    csv_return_format=read_csv_file(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    metadata_main_format['clean_train_set']['encode'] = 1
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    return jsonify(modified_train_dataset=csv_return_format, shape=train_df.shape)

@app.route('/workflow/testing/clean/encode/encode', methods=['GET', 'POST'])
def test_encode_col():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))

    encode_json = request.get_json()
    col = encode_json['col']

    test_df, flag = get_encode_cols(test_df, col)
    try:
        if flag == 1:
            return 'Could not encode'
    except:
        pass
    test_df.to_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"), index=False)
    csv_return_format=read_csv_file(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    metadata_main_format['clean_test_set']['encode'] = 1
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    return jsonify(modified_test_dataset=csv_return_format, shape=test_df.shape)




@app.route('/workflow/training/model')
def model_tab():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    if len(metadata_main_format['model'])==0:
        return jsonify({"status": "No models available", "action": "Create your first model by clicking 'Add'"})
    model_list=[]
    for md in metadata_main_format['model']:
        model_list.append('Model {}'.format(md['model_id']))
    return jsonify(current_list_of_created_models=model_list)

@app.route('/workflow/training/model/add', methods=['GET', 'POST'])
def add_model():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    if metadata_main_format['input']['training_set']==0:
        return jsonify({"status": "No dataset selected", "action": "Kindly browse the training dataset first"})
    metadata_main_format['model_id_counter'] = metadata_main_format['model_id_counter'] + 1
    metadata_model_dict = {
                    "model_id": metadata_main_format['model_id_counter'],
                    "target_variable": 0,
                    "train_validation_split": 0,
                    "re_shuffle_split": 0,
                    "algorithm": 0,
                    "normalization": 0,
                    "standardization": 0,
                    "scale_flag": 0,
                    "tune": 0,
                    "fit": 0,
                    "validate": 0,
                    "metrics": 0,
                    "publish": 0,
                    "publishing_name": 0
                 }
    metadata_main_format['model'].append(metadata_model_dict)
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    return jsonify(currently_added_model=metadata_model_dict)

@app.route('/workflow/training/model/model_selection', methods=['GET', 'POST'])
def model_selection():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    mid = request.args.get('model_id_selected')
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    for md in metadata_main_format['model']:
        if md['model_id']==int(mid):
            return jsonify(selected_model_info=md)

@app.route('/workflow/training/model/delete', methods=['GET', 'POST'])
def delete_model():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    mid = request.args.get('model_id_selected')
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    for i in range(len(metadata_main_format['model'])):
        if metadata_main_format['model'][i]['model_id'] == int(mid):
            del metadata_main_format['model'][i]
            break
    model_list=[]
    for md in metadata_main_format['model']:
        model_list.append('Model {}'.format(md['model_id']))
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    #return jsonify({"Model list after deletion":model_list, "Deleted model":"Model {}".format(mid)})
    return jsonify(Model_list_after_deletion=model_list, Deleted_model="Model {}".format(mid))

@app.route('/workflow/training/model/fit_model', methods=['GET', 'POST'])
def fit_model_and_tune_and_validate_and_evaluate():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))

    #global train_flag
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    train_flag = metadata_main_format['input']['train_flag']

    if train_flag != 1:
        return jsonify({"status": "No dataset selected", "action": "Kindly browse the dataset first"})

    mb = request.get_json()
    print('\n\n === MB ===\n\n', mb)
    # Metadata status update
    for md in metadata_main_format['model']:
        print('\n ABCD')
        if md['model_id'] == mb['model_id']:
            print('\n ===== BINGO =====')
            md['target_variable'] = mb['target_variable']
            md['train_validation_split'] = mb['train_validation_split']
            md['re_shuffle_split'] = mb['re_shuffle_split']
            md['algorithm'] = mb['algorithm']
            md['normalization'] = mb['normalization']
            md['standardization'] = mb['standardization']
            print('\n\n === MD ===\n\n', md)
            tud = mb['tune']
            print('\n\n === tud ===\n\n', tud)

    print('\n === Begin fitting process ===\n')

    # Target Selection
    X, y = target_selection(train_df, mb['target_variable'])

    # Train validation split
    if mb['train_validation_split']=='yes':
        rss = mb['re_shuffle_split']
        X_train, X_val, y_train, y_val=TVS_yes(X, y, rss) #BACKEND
        tvs_flag=1
    else:
        X_train, X_val, y_train, y_val=TVS_no(X, y) #BACKEND
        tvs_flag=0

    #global scale_flag

    # Normalization
    if mb['normalization']=='yes':
        X_train, X_val, scale_flag = normalize(username, projectname, tvs_flag, X_train, X_val, mb['model_id'])
        if scale_flag==None:
            return '\nERROR: Could not Normalize the model.\n\nTip: Ensure that the data is cleaned & transformed properly.\n\nTry again !\n'
    else:
        scale_flag=0

    # Standardization
    if mb['standardization']=='yes':
        X_train, X_val, scale_flag = standardize(username, projectname, tvs_flag, X_train, X_val, mb['model_id'])
        if scale_flag==None:
            return '\nERROR: Could not Standardize the model.\n\nTip: Ensure that the data is cleaned & transformed properly.\n\nTry again !\n'
    else:
        pass
    # scale_flag status update
    for md in metadata_main_format['model']:
        print('\n ABCD 899')
        if md['model_id'] == mb['model_id']:
            print('\n ===== BINGO 899 =====')
            md['scale_flag'] = scale_flag

    # Model fitting
    #global model
    model, tune_key = fit_model(mb['algorithm'], X_train, y_train, tud) # algo = mb['algorithm']
    print('\n\n === Model ===\n\n', model)
    if model==None:
        #DUMP Metadata JSON
        write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

        return '\nERROR: Could not fit the model.\n\nTip: Ensure that the data is cleaned or transformed properly.\n\nTry again !\n'
    else:
        # Metadata status update
        for md in metadata_main_format['model']:
            print('\n ABCD 2')
            if md['model_id'] == mb['model_id']:
                print('\n ===== BINGO 2 =====')
                md['tune'] = tud
                md['fit'] = 1
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    #return 'Model fitting successful !!!'

    # Validation process
    pred_val, pred_train, pvf = validation_pred(model, tvs_flag, X_train, X_val)
    # Saving predictions at backend
    train_folder_path = os.path.join(get_project_folder_path(username, projectname), "Training folder")
    if pvf==1:
        training_pred_folder_creation(pred_train, X_train, y_train, train_folder_path)
        validation_pred_folder_creation(pred_val, X_val, y_val, train_folder_path)
    else:
        training_pred_folder_creation(pred_train, X_train, y_train, train_folder_path)
    # Metadata status update
    for md in metadata_main_format['model']:
        print('\n ABCD 3')
        if md['model_id'] == mb['model_id']:
            print('\n ===== BINGO 3 =====')
            md['validate'] = 1
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))


    # Metric Process
    evaluation_metrics = evaluation_metrics_setup(mb['algorithm'])
    print('\n === Metrics ===\n', evaluation_metrics)
    try:

        cmtdf, cmvdf, train_metrics, val_metrics = metric_calculation(mb['model_id'], tvs_flag, evaluation_metrics, pred_val, y_val, pred_train, y_train)
    except:
        return '\nERROR: Could not evaluate the model.\n\nTip: Ensure that the data is cleaned or transformed properly.\n\nTry again !\n'
    mdf = metric_table(mb['model_id'], evaluation_metrics, train_metrics, val_metrics)
    try:
        os.mkdir(os.path.join(get_project_folder_path(username, projectname), "CSVsummary"))
    except:
        pass
    # Creating final Metric CSV for sending
    if 'Accuracy' in evaluation_metrics: # classification
        if tvs_flag==1:
            cmtdf.to_csv(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_metric_model_id_{}_(confusion_matrix_training).csv".format(mb['model_id'])), index=False)
            cmtdf_rf = read_csv_file(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_metric_model_id_{}_(confusion_matrix_training).csv".format(mb['model_id'])))
            #crf_tcm = json.dumps(cmtdf_rf, cls=DecimalEncoder)
            cmvdf.to_csv(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_metric_model_id_{}_(confusion_matrix_validation).csv".format(mb['model_id'])), index=False)
            cmvdf_rf = read_csv_file(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_metric_model_id_{}_(confusion_matrix_validation).csv".format(mb['model_id'])))
            #crf_vcm = json.dumps(cmvdf_rf, cls=DecimalEncoder)
        else:
            cmtdf.to_csv(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_metric_model_id_{}_(confusion_matrix_training).csv".format(mb['model_id'])), index=False)
            cmtdf_rf = read_csv_file(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_metric_model_id_{}_(confusion_matrix_training).csv".format(mb['model_id'])))
            #crf_tcm = json.dumps(cmtdf_rf, cls=DecimalEncoder)
            cmvdf_rf = 'No validation confusion matrix'
    else: # regression
        cmtdf_rf = 'No confusion matrix'
        cmvdf_rf = 'No confusion matrix'
    print('\n\n === Final mdf ===\n\n', mdf)

    mdf.to_csv(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_metric_model_id_{}.csv".format(mb['model_id'])), index=False)
    csv_return_format = read_csv_file(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_metric_model_id_{}.csv".format(mb['model_id'])))
    #crf = json.dumps(csv_return_format, cls=DecimalEncoder)
    # Metadata status update
    for md in metadata_main_format['model']:
        print('\n ABCD 4')
        if md['model_id'] == mb['model_id']:
            print('\n ===== BINGO 4 =====')
            md['metrics'] = 1
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))


    # Auto-publishing model for referencing and backup (Important step). to eliminate the need for global model
    try:
        os.mkdir(os.path.join(get_project_folder_path(username, projectname), "Models"))
    except:
        pass
    print('\n === Preparing model auto-publish ===\n', model)
    import pickle
    publish_name = 'Model_{}_{}.pkl'.format(mb['model_id'], scale_flag)
    pkl_file = os.path.join(get_project_folder_path(username, projectname), "Models", publish_name)
    data = model
    file = open(pkl_file, "wb")
    pickle.dump(data, file)
    file.close()
    print('\n === Actual model ===\n', model)
    print('\n === Model_{}_{} auto-published ===\n'.format(mb['model_id'], scale_flag))
    # Metadata status update
    for md in metadata_main_format['model']:
        print('\n ABCD 50')
        if md['model_id'] == mb['model_id']:
            print('\n ===== BINGO 50 =====')
            md['publishing_name'] = publish_name
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))


    print('=========TYPE CRF_OG =========\n', type(csv_return_format))
    #print('=========TYPE CRF =========\n', type(crf))
    #return jsonify({"confusion_matrix_training": cmtdf_rf, "confusion_matrix_validation": cmvdf_rf, "other_metrics": csv_return_format})
    return jsonify(confusion_matrix_training=cmtdf_rf, confusion_matrix_validation=cmvdf_rf, other_metrics=csv_return_format)

@app.route('/workflow/training/model/compare', methods=['GET', 'POST'])
def compare_model_basic_metrics():
    model_id_dict = request.args.to_dict(flat=False)
    model_id_list = model_id_dict['model_id_selected_list']
    username = model_id_dict['username']
    project_id = model_id_dict['project_id']
    project_name = 'Project {}'.format(project_id)

    print('\n\n === MODEL_ID_LIST ===\n\n', model_id_list)
    print('\n\n === MODEL_ID_LIST TYPE ===\n\n', type(model_id_list))
    final_return_dict = {}
    for model_id in model_id_list:
        print('\n === MIL ===\n', model_id)
        print('\n === TYPE MIL ===\n', type(model_id))
        try:
            csv_return_format = read_csv_file(os.path.join(get_project_folder_path(username, projectname), "CSVsummary", "current_metric_model_id_{}.csv".format(model_id)))
        except:
            csv_return_format = 'No metrics available!'
        final_return_dict["Model {}".format(model_id)] = csv_return_format
    return jsonify(final_return_dict)

@app.route('/workflow/training/model/publish', methods=['GET', 'POST'])
def publish_export_model():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)
    model_id = request.args.get('model_id')

    # Auto-importing the right model, so that it can be published (important step)
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    # Fetching
    for md in metadata_main_format['model']:
        print('\n ABCD 60')
        if md['model_id'] == int(model_id):
            print('\n ===== BINGO 60 =====')
            pub_name = md['publishing_name']
    fp = os.path.join(get_project_folder_path(username, projectname), "Models", pub_name)
    print('FP:', fp)
    import pickle
    infile = open(fp,'rb')
    model = pickle.load(infile)
    infile.close()

    #global model
    #global scale_flag
    # FETCHING scale_flag
    for md in metadata_main_format['model']:
        print('\n ABCD 999')
        if md['model_id'] == int(model_id):
            print('\n ===== BINGO 999 =====')
            scale_flag = md['scale_flag']

    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    for md in metadata_main_format['model']:
        if md['model_id']==int(model_id):
            if md['validate'] != 1:
                return 'The model selected has not yet been validated in the fitting process. Hence, cannot publish the model\n\nKindly fit the model successfully, then try to publish'
    # Model publishing format is "Model_{model_id}_{scale_flag}.pkl". Eg.: "Model_1_n1.pkl".
    publish_name = publish_model(username, projectname, model, model_id, scale_flag)
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    # Metadata status update
    for md in metadata_main_format['model']:
        print('\n ABCD 5')
        if md['model_id'] == int(model_id):
            print('\n ===== BINGO 5 =====')
            md['publish'] = 1
            md['publishing_name'] = publish_name
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    #return jsonify({"status": "Model published successfully", "publishing_name": publish_name})
    return jsonify(status="Model published successfully", publishing_name=publish_name)



@app.route('/workflow/testing/test', methods=['GET', 'POST'])
def published_model_list_display_in_test_tab():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    if len(metadata_main_format['model'])==0:
        return jsonify({"status": "No models available", "action": "Create your first model by clicking 'Add'"})
    model_list=[]
    for md in metadata_main_format['model']:
        if md['publish']==1:
            model_list.append('Model {}'.format(md['model_id']))
    return jsonify(current_list_of_previously_published_models=model_list)

@app.route('/workflow/testing/test/import', methods=['GET', 'POST'])
def import_testing_model_from_current_project():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)
    #global model
    model_id = request.args.get('model_id_selected')
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    # Fetching
    for md in metadata_main_format['model']:
        print('\n ABCD 6')
        if md['model_id'] == int(model_id):
            print('\n ===== BINGO 6 =====')
            pub_name = md['publishing_name']
    model = import_model(username, projectname, pub_name)
    # Metadata status update
    metadata_main_format['import'] = 'Model {}'.format(model_id)
    metadata_main_format['test'] = 0
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    #return jsonify({"status": "Model {} imported successfully".format(model_id), "imported_model_name": pub_name})
    return jsonify(status="Model {} imported successfully".format(model_id), imported_model_name=pub_name)

@app.route('/workflow/testing/test/test', methods=['GET', 'POST'])
def generate_test_predictions():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)
    model_id = int(request.args.get('model_id_selected'))

    #global model
    # LOAD Metadata JSON
    metadata_main_format = read_json_file(get_metadata_json_path(username,projectname))
    # Fetching
    for md in metadata_main_format['model']:
        print('\n ABCD 66')
        if md['model_id'] == int(model_id):
            print('\n ===== BINGO 66 =====')
            pub_name = md['publishing_name']
    model = import_model(username, projectname, pub_name)
    print('\n === imported model while testing ===\n', model)

    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))

    '''
    # Fetching
    for md in metadata_main_format['model']:
        print('\n ABCD 8')
        if md['model_id'] == int(model_id):
            print('\n ===== BINGO 8 =====')
            pub_name = md['publishing_name']
    scale_flag = pub_name.split('.')[0].split('_')[-1]
    '''

    # Fetching
    for md in metadata_main_format['model']:
        print('\n ABCD 88')
        if md['model_id'] == int(model_id):
            print('\n ===== BINGO 88 =====')
            scale_flag = md['scale_flag']

    print('\n === imported model scale flag ===\n', scale_flag)
    pred_test = test_pred(username, projectname, model, scale_flag, test_df, model_id)
    test_folder_path = os.path.join(get_project_folder_path(username, projectname), "Testing folder")
    testing_pred_folder_creation(pred_test, test_df, test_folder_path)
    # Metadata status update
    metadata_main_format['test'] = 1
    #DUMP Metadata JSON
    write_json_file(metadata_main_format, get_metadata_json_path(username,projectname))

    csv_return_format = read_csv_file(os.path.join(test_folder_path, "Predictions_for_TestSet.csv"))
    #crf = json.dumps(csv_return_format, cls=DecimalEncoder)
    return jsonify(generated_test_predictions=csv_return_format)




@app.route('/workflow/training/plots/line_box_contour_swarm_bar/plot', methods=['GET', 'POST'])
def train_line_box_contour_swarm_bar_plots():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))

    pj = request.get_json() # plot_json
    pt = pj['plot_type']
    x = pj['x']
    y = pj['y']
    color = pj['color']

    fig = get_line_box_contour_swarm_bar_plots(train_df, pt, x, y, color)

    if fig==1:
        return '\nERROR\n\nEnsure that the sort column has no NaN values\n'
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    print('\n === TYPE graphJSON ===\n', type(graphJSON))
    return graphJSON

@app.route('/workflow/testing/plots/line_box_contour_swarm_bar/plot', methods=['GET', 'POST'])
def test_line_box_contour_swarm_bar_plots():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))

    pj = request.get_json() # plot_json
    pt = pj['plot_type']
    x = pj['x']
    y = pj['y']
    color = pj['color']

    fig = get_line_box_contour_swarm_bar_plots(test_df, pt, x, y, color)

    if fig==1:
        return '\nERROR\n\nEnsure that the sort column has no NaN values\n'
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    print('\n === TYPE graphJSON ===\n', type(graphJSON))
    return graphJSON

@app.route('/workflow/training/plots/scatter/plot', methods=['GET', 'POST'])
def train_scatter_plot():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))

    pj = request.get_json() # plot_json
    x = pj['x']
    y = pj['y']
    color = pj['color']
    line_fit = pj['line_fit']

    fig = get_scatter_plots(train_df, x, y, color, line_fit)

    if fig==1:
        return '\nERROR\n\nEnsure that the sort column has no NaN values\n'
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    print('\n === TYPE graphJSON ===\n', type(graphJSON))
    return graphJSON

@app.route('/workflow/testing/plots/scatter/plot', methods=['GET', 'POST'])
def test_scatter_plot():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))

    pj = request.get_json() # plot_json
    x = pj['x']
    y = pj['y']
    color = pj['color']
    line_fit = pj['line_fit']

    fig = get_scatter_plots(test_df, x, y, color, line_fit)

    if fig==1:
        return '\nERROR\n\nEnsure that the sort column has no NaN values\n'
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    print('\n === TYPE graphJSON ===\n', type(graphJSON))
    return graphJSON

@app.route('/workflow/training/plots/heatmap/plot', methods=['GET', 'POST'])
def train_heatmap_plot():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))

    type = request.args.get('type')

    fig = get_heatmap_plots(train_df, type)

    if fig==1:
        return '\nERROR\n\nEnsure that the dataset has no NaN values\n'
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/workflow/testing/plots/heatmap/plot', methods=['GET', 'POST'])
def test_heatmap_plot():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))

    type = request.args.get('type')

    fig = get_heatmap_plots(test_df, type)

    if fig==1:
        return '\nERROR\n\nEnsure that the dataset has no NaN values\n'
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/workflow/training/plots/histogram/plot', methods=['GET', 'POST'])
def train_histogram_plot():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global train_df
    train_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Training folder", "train_df_updated.csv"))

    pj = request.get_json() # plot_json
    x = pj['x']
    color = pj['color']
    bins = pj['bins']

    fig = get_histogram_plots(train_df, x, color, bins)

    if fig==1:
        return '\nERROR\n\nEnsure that the sort column has no NaN values\nAlso ensure than bins is an integer value'
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    print('\n === TYPE graphJSON ===\n', type(graphJSON))
    return graphJSON

@app.route('/workflow/testing/plots/histogram/plot', methods=['GET', 'POST'])
def test_histogram_plot():
    username = request.args.get('username')
    project_id = request.args.get('project_id')
    projectname = 'Project {}'.format(project_id)

    #global test_df
    test_df = pd.read_csv(os.path.join(get_project_folder_path(username, projectname), "Testing folder", "test_df_updated.csv"))

    pj = request.get_json() # plot_json
    x = pj['x']
    color = pj['color']
    bins = pj['bins']

    fig = get_histogram_plots(test_df, x, color, bins)

    if fig==1:
        return '\nERROR\n\nEnsure that the sort column has no NaN values\n'
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    print('\n === TYPE graphJSON ===\n', type(graphJSON))
    return graphJSON

# ===== API Definitions END =====

# Execution
if __name__ == '__main__':
    app.run(debug=True)
