import numpy as np
import pandas as pd
import os
import json
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import plot
import statsmodels.api as sm #Not sure if this still has any use
import warnings
warnings.filterwarnings('ignore')

# Flow-Tracker related definitions (Metadata)
def tracker_create():
    track_dict = {
                  'input'       :'x',
                  'eda_train'   :'x',
                  'eda_test'    :'x',
                  'model'       :'x',
                  'import'      :'x',
                  'tune'        :'x',
                  'validate'    :'x',
                  'test'        :'x',
                  'export'      :'x'
                 }
    return track_dict

def tracker_id_list():
    empty_list=[]
    return empty_list.copy()

def tracker_id_dict():
    empty_dict={}
    return empty_dict.copy()

def load_tracker():
    with open(os.getcwd()+'\\Metadata.json') as file:
        track_dict=json.load(file)
    return track_dict

def dump_tracker(track_dict):
    with open(os.getcwd()+'\\Metadata.json', 'w') as file:
        json.dump(track_dict, file, indent=4)

def tracker_dict_mod(dict, key, value):
    dict[key]=value
    return dict

def tracker_add_list(list_id, list_name, list_add):
    track_dict=load_tracker()
    list_id.append(list_add)
    track_dict[list_name]=list_id
    dump_tracker(track_dict)

def tracker_add_dict(list_name, list_id, list_add):
    track_dict=load_tracker()
    if list_add not in list_id:
        list_id.append(list_add)
    track_dict[list_name]={'run_id':list_id}
    dump_tracker(track_dict)

def tracker_add_status(id):
    track_dict=load_tracker()
    track_dict[id]='yes'
    dump_tracker(track_dict)

# Definitions for reading the datasets
def read_data(fp, hc):
    try:
        read_flag=0
        if hc:
            df=pd.read_csv(fp)
            header_list=list(df.columns)
            data=df.values.tolist()
            read_flag=1
        else:
            df=pd.read_csv(fp, header=None)
            header_list=['column' + str(x) for x in range(len(df.iloc[0]))]
            df.columns=header_list
            df.drop(0, axis=0, inplace=True)
            data=df.values.tolist()
            read_flag=1
        return (df, data, header_list, read_flag)
    except:
        df=None
        header_list=None
        data=None
        return (df, data, header_list, read_flag)

def which_set_train(train_df, train_header_list, fn1):
    r1=1
    df=train_df
    data=df.values.tolist()
    header_list=train_header_list
    fn=fn1
    return (df, data, header_list, fn, r1)

def which_set_test(test_df, test_header_list, fn2):
    r1=0
    df=test_df
    data=df.values.tolist()
    header_list=test_header_list
    fn=fn2
    return (df, data, header_list, fn, r1)

# Definitions for viewing datasets and its related info
def head_show(df):
    hdata=df.head().values.tolist()
    return hdata

def tail_show(df):
    tdata=df.tail().values.tolist()
    return tdata

def describe_show(df):
    des=df.describe()
    ddata=des.values.tolist()
    header_list=list(des.columns)
    for i, d in enumerate(ddata):
        d.insert(0, list(des.index)[i])
    header_list.insert(0, 'Features')
    return (ddata, header_list)

def shape_show(df):
    shape=df.shape
    rows_with_nan = [index for index, row in df.iterrows() if row.isnull().any()]

    return (shape, rows_with_nan)

def info_show(df):
    s={'Columns': df.columns,
       'dtypes':df.dtypes.values.tolist(),
       'Null values': df.isnull().sum().values.tolist(),
       '% Null values': round((df.isnull().sum()/len(df)*100),2).values.tolist()}
    dtdf=pd.DataFrame(s)
    dtdata=dtdf.values.tolist()
    header_list=list(dtdf.columns)
    return (dtdata, header_list)

# Definitions required for cleaning operations
def drop_rd1(r1, df):
    df=df.dropna(axis=0).reset_index(drop=True)
    if r1==1:
        train_df=df.copy()
        train_header_list=list(train_df.columns)
        return (train_df, train_header_list, r1)
    else:
        test_df=df.copy()
        test_header_list=list(test_df.columns)
        return (test_df, test_header_list, r1)

def drop_rd2(r1, col, df):
    df=df.dropna(subset=[col], axis=0).reset_index(drop=True)
    if r1==1:
        train_df=df.copy()
        train_header_list=list(train_df.columns)
        return (train_df, train_header_list, r1)
    else:
        test_df=df.copy()
        test_header_list=list(test_df.columns)
        return (test_df, test_header_list, r1)

def drop_rd3(header_list, maxna, df, r1):
    tot_cols=len(header_list)
    thresh=tot_cols-int(maxna)
    df=df.dropna(axis=0, thresh=thresh).reset_index(drop=True)
    if r1==1:
        train_df=df.copy()
        train_header_list=list(train_df.columns)
        return (train_df, train_header_list, r1)
    else:
        test_df=df.copy()
        test_header_list=list(test_df.columns)
        return (test_df, test_header_list, r1)

def drop_rd4(r1, i, df):
    df=df.drop(index=int(i), axis=0).reset_index(drop=True)
    if r1==1:
        train_df=df.copy()
        train_header_list=list(train_df.columns)
        return (train_df, train_header_list, r1)
    else:
        test_df=df.copy()
        test_header_list=list(test_df.columns)
        return (test_df, test_header_list, r1)

def drop_column(r1, dcol, df):
    df=df.drop(dcol, axis=1)
    if r1==1:
        train_df=df.copy()
        train_header_list=list(train_df.columns).copy()
        return (train_df, train_header_list, r1)
    else:
        test_df=df.copy()
        test_header_list=list(test_df.columns).copy()
        return (test_df, test_header_list, r1)

def num_cat_cols(df):
    num_cols=[col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    cat_cols=[col for col in df.columns if df[col].dtype == 'object']
    return (num_cols, cat_cols)

def replace_ncr1(sel_num_col, gt, nv, df):
    df[sel_num_col].values[df[sel_num_col].values>int(gt)]=int(nv)

def replace_ncr2(sel_num_col, lt, nv, df):
    df[sel_num_col].values[df[sel_num_col].values<int(lt)]=int(nv)

def replace_ncr3(sel_num_col, uv, nv, df):
    df[sel_num_col]=df[sel_num_col].replace([int(uv)], int(nv))

def replace_cat(sel_cat_col, ov, nv, df):
    df[sel_cat_col]=df[sel_cat_col].replace([ov], nv)

def fill_num_mean(df, sel_num_col):
    if sel_num_col=='All numerical columns':
        num_cols, cat_cols =num_cat_cols(df)
        df[num_cols]=df[num_cols].fillna(value=df[num_cols].mean())
    else:
        df[sel_num_col]=df[sel_num_col].fillna(value=df[sel_num_col].mean())

def fill_num_median(df, sel_num_col):
    if sel_num_col=='All numerical columns':
        num_cols, cat_cols =num_cat_cols(df)
        df[num_cols]=df[num_cols].fillna(value=df[num_cols].median())
    else:
        df[sel_num_col]=df[sel_num_col].fillna(value=df[sel_num_col].median())

def fill_num_mode(df, sel_num_col):
    if sel_num_col=='All numerical columns':
        num_cols, cat_cols =num_cat_cols(df)
        df[num_cols]=df[num_cols].fillna(value=df[num_cols].mode().iloc[0], axis=0)
    else:
        df[sel_num_col]=df[sel_num_col].fillna(value=df[sel_num_col].mode().iloc[0], axis=0)

def fill_num_interpolation(df, sel_num_col):
    if sel_num_col=='All numerical columns':
        num_cols, cat_cols =num_cat_cols(df)
        df[num_cols]=df[num_cols].interpolate(method ='linear', limit_direction ='both')
    else:
        df[sel_num_col]=df[sel_num_col].interpolate(method ='linear', limit_direction ='both')

def fill_num_previous(df, sel_num_col):
    if sel_num_col=='All numerical columns':
        num_cols, cat_cols =num_cat_cols(df)
        df[num_cols]=df[num_cols].interpolate(method ='pad')
    else:
        df[sel_num_col]=df[sel_num_col].interpolate(method ='pad')

def fill_num_unique(df, uv, sel_num_col):
    if sel_num_col=='All numerical columns':
        num_cols, cat_cols =num_cat_cols(df)
        df[num_cols]=df[num_cols].fillna(value=int(uv))
    else:
        df[sel_num_col]=df[sel_num_col].fillna(value=int(uv))

def fill_cat_mf(df, sel_cat_col):
    if sel_cat_col=='All categorical columns':
        num_cols, cat_cols =num_cat_cols(df)
        df[cat_cols]=df[cat_cols].fillna(value=df[cat_cols].mode().iloc[0], axis=0)
    else:
        df[sel_cat_col]=df[sel_cat_col].fillna(value=df[sel_cat_col].mode().iloc[0], axis=0)

def fill_cat_uv(df, uv, sel_cat_col):
        if sel_cat_col=='All categorical columns':
            num_cols, cat_cols =num_cat_cols(df)
            df[cat_cols]=df[cat_cols].fillna(value=uv)
        else:
            df[sel_cat_col]=df[sel_cat_col].fillna(value=uv)

def dtype_list():
    dtype_list=['int64', 'float64', 'object']
    return dtype_list

def convert_dtype(df, col, dtype):
    try:
        df[col]=df[col].astype(dtype)
    except:
        return 1

def unique_sel_window(df, col): # DISCONTINUED
    unique=df[col].unique()
    return unique

def value_counts_window(df, col):
    vc=df[col].value_counts()
    #return vc
    unique=unique_sel_window(df, col)
    unique=unique.tolist()
    if np.nan in unique:
        unique.remove(np.nan)
    s={'Unique values':unique,
       'Unique value count':vc.values.tolist()}
    vcdf=pd.DataFrame(s)
    vcdt=vcdf.values.tolist()
    vchl=list(vcdf.columns)
    return (vcdt, vchl)

def encode_window(df, col):
    if col=='All categorical columns':
        num_cols, cat_cols =num_cat_cols(df)
        for acols in cat_cols:
            df[acols]=df[acols].astype('category').cat.codes.astype('int64')
    else:
        df[col]=df[col].astype('category').cat.codes.astype('int64')


# Definitions for folder creations at different stages
def main_folder_creation(cwd):
    print('B: ',os.getcwd())
    print('C: ', cwd)
    try:
        os.mkdir('Dataset_journey') #dj
    except:
        pass
    dj_path=cwd + '\\Dataset_journey'
    os.chdir('Dataset_journey')
    on_date_folder=datetime.datetime.now().strftime('on %d-%m-%Y_%H-%M-%S')
    os.mkdir(on_date_folder)
    os.chdir(on_date_folder)
    print(os.getcwd())
    return (dj_path, on_date_folder)

def training_data_folder_creation(train_df):
    os.mkdir('For Training Dataset')
    train_folder_path=os.getcwd() + '\\For Training Dataset'
    print(train_folder_path)
    train_df.to_csv(train_folder_path+'\\Original_Training_Dataset (as input).csv', index=False)
    return(train_folder_path)

def testing_data_folder_creation(test_df):
    os.mkdir('For Testing Dataset')
    test_folder_path=os.getcwd() + '\\For Testing Dataset'
    print(test_folder_path)
    test_df.to_csv(test_folder_path+'\\Original_Testing_Dataset (as input).csv', index=False)
    return(test_folder_path)

def after_drop_folder_updation_train(train_folder_path, train_df):
    train_df.to_csv(train_folder_path+'\\Training_Dataset_after_drop_operations.csv', index=False)

def after_drop_folder_updation_test(test_folder_path, test_df):
    test_df.to_csv(test_folder_path+'\\Testing_Dataset_after_drop_operations.csv', index=False)

def after_replace_folder_updation_train(train_folder_path, train_df):
    train_df.to_csv(train_folder_path+'\\Training_Dataset_after_replace_operations.csv', index=False)

def after_replace_folder_updation_test(test_folder_path, test_df):
    test_df.to_csv(test_folder_path+'\\Testing_Dataset_after_replace_operations.csv', index=False)

def after_fillna_folder_updation_train(train_folder_path, train_df):
    train_df.to_csv(train_folder_path+'\\Training_Dataset_after_fillNaN_operations.csv', index=False)

def after_fillna_folder_updation_test(test_folder_path, test_df):
    test_df.to_csv(test_folder_path+'\\Testing_Dataset_after_fillNaN_operations.csv', index=False)

def after_convert_folder_updation_train(train_folder_path, train_df):
    train_df.to_csv(train_folder_path+'\\Training_Dataset_after_Dtype_conversion_operations.csv', index=False)

def after_convert_folder_updation_test(test_folder_path, test_df):
    test_df.to_csv(test_folder_path+'\\Testing_Dataset_after_Dtype_conversion_operations.csv', index=False)

def training_pred_folder_creation(pred_train, X_train, y_train, train_folder_path):
    output_train=X_train.reset_index().copy()
    train_predictions=pd.DataFrame(pred_train)
    train_predictions.columns=['Predicted_target']
    train_target=y_train.reset_index(drop=True).copy()
    #val_target.columns=['Actual_target']
    output_train=pd.concat([output_train, train_predictions], axis=1)
    output_train=pd.concat([output_train, train_target], axis=1)
    output_train.to_csv(train_folder_path+'\\Predictions_for_TrainSet.csv', index=False)

def validation_pred_folder_creation(pred_val, X_val, y_val, train_folder_path):
    output_val=X_val.reset_index().copy()
    val_predictions=pd.DataFrame(pred_val)
    val_predictions.columns=['Predicted_target']
    val_target=y_val.reset_index(drop=True).copy()
    #val_target.columns=['Actual_target']
    output_val=pd.concat([output_val, val_predictions], axis=1)
    output_val=pd.concat([output_val, val_target], axis=1)
    output_val.to_csv(train_folder_path+'\\Predictions_for_ValidationSet.csv', index=False)

def after_encode_folder_updation_train(train_folder_path, train_df):
    train_df.to_csv(train_folder_path+'\\Training_Dataset_after_Label_Encoding_operations.csv', index=False)

def after_encode_folder_updation_test(test_folder_path, test_df):
    test_df.to_csv(test_folder_path+'\\Testing_Dataset_after_Label_Encoding_operations.csv', index=False)

def testing_pred_folder_creation(pred_test, X_test, test_folder_path):
    output_test=X_test.reset_index().copy()
    test_predictions=pd.DataFrame(pred_test)
    test_predictions.columns=['Predicted_target']
    #val_target.columns=['Actual_target']
    output_test=pd.concat([output_test, test_predictions], axis=1)
    output_test.to_csv(test_folder_path+'\\Predictions_for_TestSet.csv', index=False)



# Definitions required for plots
def line_plot(df, x, y, hue):
    if x=='Over index':
        # SEABORN CODE
        # Disfigured
        #plt.figure(figsize=(12,5))
        #sns.lineplot(data=df)
        #plt.title('Lineplot for all numerical columns')
        #plt.show(block=False)
        #return None

        # PLOTLY CODE
        try:
            fig = px.line(data_frame=df, x=df.index, y=y, color=hue)
        except:
            return 1
        fig.update_layout(title='Line plot: "{}" across dataset'.format(y))
        plot(fig)

    else:
        # SEABORN CODE
        #plt.figure(figsize=(12,5))
        #sns.lineplot(data=df[line_for_col], label=line_for_col)
        #plt.title('Lineplot for {} column'.format(line_for_col))
        #plt.show(block=False)
        #return None

        # PLOTLY CODE
        try:
            fig = px.line(data_frame=df, x=x, y=y, color=hue)
        except:
            return 1
        fig.update_layout(title='Line plot: "{}" vs "{}"'.format(y, x))
        plot(fig)

def index_column_list(df):
    index_list=df.index.tolist()
    index_list.insert(0, 'Complete Index')
    column_list=df.columns.tolist()
    column_list.insert(0, 'Complete Column')
    return(index_list, column_list)

#def bar_plot(df, x_par, y_par, hue, fx, fy):
def bar_plot(df, x, y, hue):
    '''if fx==1:
        if x_par=='Complete Index':
            x=df.index
        else:
            x=df.loc[x_par]
    else: # fx==2
        if x_par=='Complete Column':
            x=df.columns
        else:
            #x=df[x_par]
            x=x_par
    if fy==1:
        if y_par=='Complete Index':
            y=df.index
        else:
            y=df.loc[y_par]
    else: # fy==2
        if y_par=='Complete Column':
            y=df.columns
        else:
            #y=df[y_par]
            y=y_par
    if hue==None:
        pass
    else:
        #hue=df[hue]
        pass'''
    try:
        #sns.barplot(x=x, y=y, hue=hue)
        fig=px.bar(data_frame=df, x=x, y=y, color=hue)
    except:
        return 1
    #plt.title('Bar plot for {} vs {} columns'.format(y, x))
    #plt.show(block=False)
    fig.update_layout(title='Bar plot for {} vs {} columns'.format(y, x))
    plot(fig)

def heatmap_plot(fhm, df):
    hm_error=0
    if fhm==1:
        try:
            #sns.heatmap(data=df, annot=True, linewidth=0.2, cbar_kws={"shrink": .9})
            fig=px.imshow(df)
        except:
            hm_error=1
            return hm_error
        #plt.title('Heatmap for complete Dataset')
        #plt.show(block=False)
        fig.update_layout(title='Heatmap for complete Dataset')
        plot(fig)
    else:
        try:
            corr=df.corr()
        except:
            hm_error=1
            return hm_error
        mask = np.triu(np.ones_like(corr, dtype=np.bool))
        mask=mask[1: , :-1]
        corr=corr.iloc[1: , :-1]
        cmap = sns.diverging_palette(240, 10, n=9)
        try:
            #sns.heatmap(data=corr, cmap=cmap, annot=True, mask=mask, vmin=-1, vmax=1, linewidth=0.2, square=True, cbar_kws={"shrink": .9})
            fig=px.imshow(corr)
        except:
            hm_error=1
            return hm_error
        #plt.yticks(rotation=0)
        #plt.xticks(rotation=45)
        #plt.title('Correlation Heatmap')
        #plt.show(block=False)
        fig.update_layout(title='Correlation Heatmap')
        plot(fig)

def scatter_plot(df, x, y, hue, skr):
    if skr: # Keep Reg Line

        # SEABORN CODE
        #sns.lmplot(x=x, y=y, hue=hue, data=df)
        #plt.title('Scatter of {} vs {} columns (with regression line)'.format(y, x))
        #plt.show(block=False)

        # PLOTLY CODE
        try:
            fig = px.scatter(data_frame=df, x=x, y=y, trendline='ols', color=hue)
        except:
            return 1
        fig.update_layout(title='Scatter plot of "{}" vs "{}" columns'.format(y, x))
        plot(fig)

    else: # No Reg Line
        if hue==None:
            pass
        else:
            #hue=df[hue]    # For SEABORN CODE
            hue=hue        # For PLOTLY CODE

        # SEABORN CODE
        #sns.scatterplot(x=df[x], y=df[y], hue=hue)
        #plt.title('Scatter of {} vs {} columns'.format(y, x))
        #plt.show(block=False)

        # PLOTLY CODE
        try:
            fig = px.scatter(data_frame=df, x=x, y=y, color=hue)
        except:
            return 1
        fig.update_layout(title='Scatter plot of "{}" vs "{}" columns'.format(y, x))
        plot(fig)

def swarm_plot(df, x, y, hue):
    if hue==None:
        pass
    else:
        pass#hue=df[hue]
    #sns.swarmplot(x=df[x], y=df[y], hue=hue, dodge=True)
    #plt.title('Swarm plot of {} vs {} columns'.format(y, x))
    #plt.show(block=False)
    try:
        fig=px.strip(df, x=x, y=y, color=hue)
    except:
        return 1
    fig.update_layout(title='Swarm plot of {} vs {} columns'.format(y, x))
    plot(fig)

def box_plot(df, x, y, hue):
    #sns.boxplot(x=x, y=y, hue=hue, data=df, width=0.5)
    #plt.title('Box plot of {} vs {} columns'.format(y, x))
    #plt.show(block=False)
    try:
        fig=px.box(df, x=x, y=y, color=hue)
    except:
        return 1
    fig.update_layout(title='Box plot of {} vs {} columns'.format(y, x))
    plot(fig)

def hist_plot(df, x, hue):
    #sns.histplot(x=x, data=df, hue=hue)
    #plt.title('Histogram of {} column'.format(x))
    #plt.show(block=False)
    try:
        fig=px.histogram(df, x=x, color=hue, barmode='group')
    except:
        return 1
    fig.update_layout(title='Histogram of {} column'.format(x))
    plot(fig)

def density_plot(df, x, hue): #DISCONTINUED
    sns.kdeplot(x=x, data=df, hue=hue, shade=True)
    plt.title('Density plot of {} column'.format(x))
    plt.show(block=False)

def join_plot(df, x, y, hue): # Density Contour
    #sns.jointplot(x=x, y=y, hue=hue, data=df, kind='kde', shade=True)
    #plt.title('2D KDE plot of {} vs {} columns'.format(y, x))
    #plt.show(block=False)
    try:
        fig=px.density_contour(df, x=x, y=y, color=hue, marginal_x="histogram", marginal_y="histogram")
    except:
        return 1
    fig.update_layout(title='Density Contour plot of {} vs {} columns'.format(y, x))
    plot(fig)

def count_plot(df, x, hue): #DISCONTINUED
    sns.countplot(x=x, data=df, hue=hue)
    plt.title('Count plot of {} column'.format(x))
    plt.show(block=False)

# Definitions for fitting and prediction operation
def algo_list():
    algorithms=['Logistic Regression', 'KNN Classifier', 'SVM', 'Decision Tree Classifier', 'Random Forest Classifier',
                'Linear Regression', 'KNN Regressor', 'SVR', 'Decision Tree Regressor', 'Random Forest Regressor']
    return algorithms

def target_selection(df, target_col):
    y=df[target_col]
    X=df.drop(target_col, axis=1)
    return (X, y)

def TVS_yes(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val=train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    return (X_train, X_val, y_train, y_val)

def TVS_no(X, y):
    X_train=X.copy()
    y_train=y.copy()
    X_val=None
    y_val=None
    return (X_train, X_val, y_train, y_val)

def normalize(tvs_flag, X_train, X_val):
    from sklearn.preprocessing import MinMaxScaler
    norm=MinMaxScaler()

    if tvs_flag==1:
        try:
            X_train_norm = pd.DataFrame(norm.fit_transform(X_train))
        except:
            return (None, None, None)

        # saving this scaler here so that we can apply it to test set later
        import pickle
        pkl_file = os.getcwd()+ '\\For Training Dataset\\scaler.pkl' # change this to your desired folder & file name
        data = norm # define what you want to save here
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
        pkl_file = os.getcwd()+ '\\For Training Dataset\\scaler.pkl' # change this to your desired folder & file name
        data = norm # define what you want to save here
        file = open(pkl_file,"wb")
        pickle.dump(data,file)
        file.close()

        X_val_norm = None
        X_train_norm.columns = X_train.columns
        scale_flag='n0'
    return (X_train_norm, X_val_norm, scale_flag)

def standardize(tvs_flag, X_train, X_val):
    from sklearn.preprocessing import StandardScaler
    norm=StandardScaler()

    if tvs_flag==1:
        try:
            X_train_norm = pd.DataFrame(norm.fit_transform(X_train))
        except:
            return (None, None, None)

        import pickle
        pkl_file = os.getcwd()+ '\\For Training Dataset\\scaler.pkl' # change this to your desired folder & file name
        data = norm # define what you want to save here
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
        pkl_file = os.getcwd()+ '\\For Training Dataset\\scaler.pkl' # change this to your desired folder & file name
        data = norm # define what you want to save here
        file = open(pkl_file,"wb")
        pickle.dump(data,file)
        file.close()

        X_val_norm = None
        X_train_norm.columns = X_train.columns
        scale_flag='s0'
    return (X_train_norm, X_val_norm, scale_flag)

def fit_model(algo, X_train, y_train):
    print('Primary fitting IN')
    if algo==None:
        return (None, None)

    if algo=='Logistic Regression':
        #print('Logistic Regression IN')
        tune_key=1 # flag required while tuning
        from sklearn.linear_model import LogisticRegression
        lor=LogisticRegression(random_state=1)
        try:
            model=lor.fit(X_train, y_train)
        except:
            print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
            return (None, None)
        return (model, tune_key)

    if algo=='KNN Classifier':
        print('KNN Classifier IN')
        tune_key=2
        from sklearn.neighbors import KNeighborsClassifier as KNN
        knn=KNN()
        try:
            model=knn.fit(X_train, y_train)
        except:
            print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
            return (None, None)
        return (model, tune_key)

    if algo=='SVM':
        print('SVM IN')
        tune_key=3
        from sklearn import svm
        svc=svm.SVC()
        try:
            model=svc.fit(X_train, y_train)
        except:
            print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
            return None
        return (model, tune_key)

    if algo=='Decision Tree Classifier':
        print('DTC IN')
        tune_key=4
        from sklearn.tree import DecisionTreeClassifier
        dt=DecisionTreeClassifier(random_state=1)
        try:
            model=dt.fit(X_train, y_train)
        except:
            print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
            return (None, None)
        return (model, tune_key)

    if algo=='Random Forest Classifier':
        print('RFC IN')
        tune_key=5
        from sklearn.ensemble import RandomForestClassifier
        rf=RandomForestClassifier(random_state=1)
        try:
            model=rf.fit(X_train, y_train)
        except:
            print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
            return (None, None)
        return (model, tune_key)

    if algo=='Linear Regression':
        print('Linear Regression IN')
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
        print('KNN Regression IN')
        tune_key=7
        from sklearn.neighbors import KNeighborsRegressor as KNN
        print('fgh')
        knn=KNN()
        try:
            model=knn.fit(X_train, y_train)
        except:
            print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
            return (None, None)
        return (model, tune_key)

    if algo=='SVR':
        print('SVR IN')
        tune_key=8
        from sklearn.svm import SVR
        svr=SVR()
        try:
            model=svr.fit(X_train, y_train)
        except:
            print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
            return (None, None)
        return (model, tune_key)

    if algo=='Decision Tree Regressor':
        print('DTR IN')
        tune_key=9
        from sklearn.tree import DecisionTreeRegressor
        dt=DecisionTreeRegressor(random_state=1)
        try:
            model=dt.fit(X_train, y_train)
        except:
            print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
            return (None, None)
        return (model, tune_key)

    if algo=='Random Forest Regressor':
        print('RFR IN')
        tune_key=10
        from sklearn.ensemble import RandomForestRegressor
        rf=RandomForestRegressor(random_state=1)
        try:
            model=rf.fit(X_train, y_train)
        except:
            print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
            return (None, None)
        return (model, tune_key)

############################################
# Definitions for TUNING of algorithms START
# For Classification
def lor_penalty(): #parameter selection
    penalty=['l1', 'l2', 'elasticnet', 'none']
    return penalty
def lor_solver(): #parameter selection
    solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    return solver

def tune_lor(tune_key, penalty_val, C_val, solver_val, X_train, y_train):
    fit_flag=0
    from sklearn.linear_model import LogisticRegression
    try:
        lor=LogisticRegression(penalty=penalty_val, C=C_val, solver=solver_val, random_state=1)
        model=lor.fit(X_train, y_train)
        fit_flag=1
    except:
        print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
        return (None, fit_flag)
    return (model, fit_flag)

def tune_knn_c(nn, X_train, y_train):
    fit_flag=0
    from sklearn.neighbors import KNeighborsClassifier as KNN
    try:
        knn=KNN(n_neighbors=nn)
        model=knn.fit(X_train, y_train)
        fit_flag=1
    except:
        print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
        return (None, fit_flag)
    return (model, fit_flag)

def svm_kernel(): #parameter selection
    kernel=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    return kernel

def tune_svm(kernel_val, C_val, degree_val, gamma_val ,X_train, y_train):
    fit_flag=0
    from sklearn import svm
    try:
        svc=svm.SVC(kernel=kernel_val, degree=degree_val, C=C_val, gamma=gamma_val)
        model=svc.fit(X_train, y_train)
        fit_flag=1
    except:
        print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
        return (None, fit_flag)
    return (model, fit_flag)

def tune_dtc(mss_val, msl_val, md_val, mln_val ,X_train, y_train):
    fit_flag=0
    from sklearn.tree import DecisionTreeClassifier
    try:
        dt=DecisionTreeClassifier(max_depth=md_val, min_samples_leaf=msl_val, min_samples_split=mss_val, max_leaf_nodes=mln_val, random_state=1)
        model=dt.fit(X_train, y_train)
        fit_flag=1
    except:
        print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
        return (None, fit_flag)
    return (model, fit_flag)

def tune_rfc(ne_val, mss_val, msl_val, md_val, mln_val ,X_train, y_train):
    fit_flag=0
    from sklearn.ensemble import RandomForestClassifier
    try:
        rf=RandomForestClassifier(n_estimators=ne_val, max_depth=md_val, min_samples_leaf=msl_val, min_samples_split=mss_val, max_leaf_nodes=mln_val, random_state=1)
        model=rf.fit(X_train, y_train)
        fit_flag=1
    except:
        print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
        return (None, fit_flag)
    return (model, fit_flag)

# For Regression
def tune_knn_r(nn, X_train, y_train):
    fit_flag=0
    from sklearn.neighbors import KNeighborsRegressor as KNN
    try:
        knn=KNN(n_neighbors=nn)
        model=knn.fit(X_train, y_train)
        fit_flag=1
    except:
        print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
        return (None, fit_flag)
    return (model, fit_flag)

def tune_svr(kernel_val, epsilon_val, C_val, degree_val, gamma_val ,X_train, y_train):
    fit_flag=0
    from sklearn.svm import SVR
    try:
        svr=SVR(kernel=kernel_val, epsilon=epsilon_val, degree=degree_val, C=C_val, gamma=gamma_val)
        model=svr.fit(X_train, y_train)
        fit_flag=1
    except:
        print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
        return (None, fit_flag)
    return (model, fit_flag)

def tune_dtr(mss_val, msl_val, md_val, mln_val ,X_train, y_train):
    fit_flag=0
    from sklearn.tree import DecisionTreeRegressor
    try:
        dt=DecisionTreeRegressor(max_depth=md_val, min_samples_leaf=msl_val, min_samples_split=mss_val, max_leaf_nodes=mln_val, random_state=1)
        model=dt.fit(X_train, y_train)
        fit_flag=1
    except:
        print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
        return (None, fit_flag)
    return (model, fit_flag)

def tune_rfr(ne_val, mss_val, msl_val, md_val, mln_val ,X_train, y_train):
    fit_flag=0
    from sklearn.ensemble import RandomForestRegressor
    try:
        rf=RandomForestRegressor(n_estimators=ne_val, max_depth=md_val, min_samples_leaf=msl_val, min_samples_split=mss_val, max_leaf_nodes=mln_val, random_state=1)
        model=rf.fit(X_train, y_train)
        fit_flag=1
    except:
        print('\nERROR: Could not fit the model. Tip: Ensure that the data is cleaned or transformed properly')
        return (None, fit_flag)
    return (model, fit_flag)
# Definitions for TUNING of algorithms END
##########################################


def validation_pred(model, tvs_flag, tune_key, X_val, X_train, dvfSVR):
    if tvs_flag==1:
        pred_train=model.predict(X_train)
        pred_val=model.predict(X_val)
        pvf=1
    else:
        pred_train=model.predict(X_train)
        pred_val=None
        pvf=0
    return (pred_val, pred_train, pvf)

# Definitions for metrics related operations
def classification_metrics():
    classification_metrics=[
                            'Confusion matrix', 'Accuracy', 'Precision & Recall',
                            'F1-score', 'AUC-ROC', 'Log-Loss'
                           ]
    return classification_metrics

def regression_metrics():
    regression_metrics=[
                        'MAE', 'MSE', 'RMSE', 'RMSLE', 'R2', 'Adjusted R2'
                       ]
    return regression_metrics

def pr_df(pr_train, pr_val):
    prdf=pd.DataFrame({'Training':pr_train, 'Validation':pr_val}, index=['Precision', 'Recall'])
    prdata=prdf.values.tolist()
    header_list=list(prdf.columns)
    return (prdata, header_list)

def metric_scoring(tvs_flag, metric, model, pred_val, X_val, y_val, pred_train, X_train, y_train):
    if metric=='Confusion matrix':
        mf='c1' # metric flag
        #from sklearn.metrics import confusion_matrix
        #matrix=confusion_matrix(pred_val, y_val)
        #print('Confusion matrix:\n', matrix, type(matrix))
        #return (matrix, mf)
        from sklearn.metrics import plot_confusion_matrix
        if tvs_flag==1:
            fig=plt.figure(figsize=(13, 5))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.set_title('Confusion Matrix for Training set')
            ax2.set_title('Confusion Matrix for Validation set')
            tcm=plot_confusion_matrix(model, X_train, y_train, display_labels=y_train.unique(), cmap=plt.cm.Blues, ax=ax1)
            vcm=plot_confusion_matrix(model, X_val, y_val, display_labels=y_val.unique(), cmap=plt.cm.Blues, ax=ax2)
        else:
            tcm=plot_confusion_matrix(model, X_train, y_train, display_labels=y_train.unique(), cmap=plt.cm.Blues)
        plt.show(block=False)
        return (None, None, None)

    if metric=='Accuracy':
        mf='c2'
        from sklearn.metrics import accuracy_score as acc
        if tvs_flag==1:
            acc_train=acc(pred_train, y_train)
            acc_val=acc(pred_val, y_val)
            print('Train Accuracy:', acc_train, '\nValidation Accuracy:', acc_val)
        else:
            acc_train=acc(pred_train, y_train)
            acc_val=None
            print('Train Accuracy:', acc_train)
        return(acc_train, acc_val, mf)

    if metric=='Precision & Recall':
        mf='c3'
        from sklearn.metrics import precision_score as ps
        from sklearn.metrics import recall_score as rs
        if tvs_flag==1:
            ps_train=ps(pred_train, y_train, average='weighted')
            rs_train=rs(pred_train, y_train, average='weighted')

            ps_val=ps(pred_val, y_val, average='weighted')
            rs_val=rs(pred_val, y_val, average='weighted')

            pr_train=[ps_train, rs_train]
            pr_val=[ps_val, rs_val]
        else:
            ps_train=ps(pred_train, y_train, average='weighted')
            rs_train=rs(pred_train, y_train, average='weighted')

            ps_val=None
            rs_val=None

            pr_train=[ps_train, rs_train]
            pr_val=[ps_val, rs_val]
        return (pr_train, pr_val, mf)

    if metric=='F1-score':
        mf='c4'
        from sklearn.metrics import f1_score as fs
        if tvs_flag==1:
            for i in ['Y', 1]: # SETTING "pos_label" value
                try:
                    fs_train=fs(pred_train, y_train, pos_label=i)
                    fs_val=fs(pred_val, y_val, pos_label=i)
                    break
                except:
                    continue
            print('F1-score for training set:', fs_train, '\nF1-score for validation set:', fs_val)
        else:
            for i in ['Y', 1]:
                try:
                    fs_train=fs(pred_train, y_train, pos_label=i)
                    break
                except:
                    continue
            fs_val=None
            print('F1-score for training set:', fs_train, '\nF1-score for validation set:', fs_val)
        return (fs_train, fs_val, mf)

    if metric=='AUC-ROC':
        mf='c5'
        from sklearn.metrics import roc_auc_score as ras
        if tvs_flag==1:
            ras_train=ras(pred_train, y_train,  average='weighted', multi_class='ovr')
            ras_val=ras(pred_val, y_val,  average='weighted', multi_class='ovr')
            print('AUC-ROC for training set:', ras_train, '\nAUC-ROC for validation set:', ras_val)
        else:
            ras_train=ras(pred_train, y_train,  average='weighted', multi_class='ovr')
            ras_val=None
            print('AUC-ROC for training set:', ras_train, '\nAUC-ROC for validation set:', ras_val)
        return (ras_train, ras_val, mf)

    if metric=='Log-Loss':
        mf='c6'
        from sklearn.metrics import log_loss as ll
        if tvs_flag==1:
            ll_train=ll(pred_train, y_train)
            ll_val=ll(pred_val, y_val)
            print('Log-loss score for training set:', ll_train, '\nLog-loss score for validation set:', ll_val)
        else:
            ll_train=ll(pred_train, y_train)
            ll_val=None
            print('Log-loss score for training set:', ll_train, '\nLog-loss score for validation set:', ll_val)
        return (ll_train, ll_val, mf)

    if metric=='MAE':
        mf='r1'
        from sklearn.metrics import mean_absolute_error as mae
        if tvs_flag==1:
            mae_train=mae(pred_train, y_train)
            mae_val=mae(pred_val, y_val)
            print('MAE for training set:', mae_train, '\nMAE for validation set:', mae_val)
        else:
            mae_train=mae(pred_train, y_train)
            mae_val=None
            print('MAE for training set:', mae_train, '\nMAE for validation set:', mae_val)
        return (mae_train, mae_val, mf)

    if metric=='MSE':
        mf='r2'
        from sklearn.metrics import mean_squared_error as mse
        if tvs_flag==1:
            mse_train=mse(pred_train, y_train)
            mse_val=mse(pred_val, y_val)
            print('MSE for training set:', mse_train, '\nMSE for validation set:', mse_val)
        else:
            mse_train=mse(pred_train, y_train)
            mse_val=None
            print('MSE for training set:', mse_train, '\nMSE for validation set:', mse_val)
        return (mse_train, mse_val, mf)

    if metric=='RMSE':
        mf='r3'
        from sklearn.metrics import mean_squared_error as rmse
        if tvs_flag==1:
            rmse_train=rmse(pred_train, y_train, squared=False)
            rmse_val=rmse(pred_val, y_val, squared=False)
            print('RMSE for training set:', rmse_train, '\nRMSE for validation set:', rmse_val)
        else:
            rmse_train=rmse(pred_train, y_train, squared=False)
            rmse_val=None
            print('RMSE for training set:', rmse_train, '\nRMSE for validation set:', rmse_val)
        return (rmse_train, rmse_val, mf)

    if metric=='RMSLE':
        mf='r4'
        from sklearn.metrics import mean_squared_log_error as msle
        from math import sqrt
        sqrt=sqrt()
        if tvs_flag==1:
            rmsle_train=sqrt(msle(pred_train, y_train))
            rmsle_val=sqrt(msle(pred_val, y_val))
            print('RMSLE for training set:', rmsle_train, '\nRMSLE for validation set:', rmsle_val)
        else:
            rmsle_train=sqrt(msle(pred_train, y_train))
            rmsle_val=None
            print('RMSLE for training set:', rmsle_train, '\nRMSLE for validation set:', rmsle_val)
        return (rmsle_train, rmsle_val, mf)

    if metric=='R2':
        mf='r5'
        from sklearn.metrics import r2_score as r2
        if tvs_flag==1:
            r2_train=r2(pred_train, y_train)
            r2_val=r2(pred_val, y_val)
            print('R2 score for training set:', r2_train, '\nR2 score for validation set:', r2_val)
        else:
            r2_train=r2(pred_train, y_train)
            r2_val=None
            print('R2 score for training set:', r2_train, '\nR2 score for validation set:', r2_val)
        return (r2_train, r2_val, mf)

    if metric=='Adjusted R2':
        mf='r6'
        from sklearn.metrics import r2_score as r2
        if tvs_flag==1:
            r2_train=r2(pred_train, y_train)
            r2_val=r2(pred_val, y_val)

            ar2_train=(1-(1-r2_train))*((len(X_train)-1)/(len(X_train)-len(X_train.columns)-1))
            ar2_val=(1-(1-r2_val))*((len(X_val)-1)/(len(X_val)-len(X_val.columns)-1))
            print('Adjusted R2 score for training set:', ar2_train, '\nAdjusted R2 score for validation set:', ar2_val)
        else:
            r2_train=r2(pred_train, y_train)
            ar2_train=(1-(1-r2_train))*((len(X_train)-1)/(len(X_train)-len(X_train.columns)-1))
            ar2_val=None
            print('Adjusted R2 score for training set:', ar2_train, '\nAdjusted R2 score for validation set:', ar2_val)
        return (ar2_train, ar2_val, mf)

def test_pred(model, scale_flag, fp, X_test, imp_flag):
    if imp_flag==1 and scale_flag!=0: # imported & scaled
        path_list=fp.split('/')
        path_list=path_list[:-1]
        new_fp='\\'.join(path_list)
        print('new_fp', new_fp)
        scaler_path = new_fp + '\\For Training Dataset\\scaler.pkl'
        import pickle
        infile = open(scaler_path,'rb')
        norm = pickle.load(infile)
        infile.close()
        X_test_norm = pd.DataFrame(norm.transform(X_test))
        X_test_norm.columns = X_test.columns
        X_test=X_test_norm.copy()


    elif imp_flag==0 and (scale_flag=='n1' or scale_flag=='n0'):
        import pickle
        fp=os.getcwd()+ '\\For Training Dataset\\scaler.pkl'
        infile = open(fp,'rb')
        norm = pickle.load(infile)
        infile.close()
        X_test_norm = pd.DataFrame(norm.transform(X_test))
        X_test_norm.columns = X_test.columns
        X_test=X_test_norm.copy()

    elif imp_flag==0 and (scale_flag=='s1' or scale_flag=='s0'):
        import pickle
        fp=os.getcwd()+ '\\For Training Dataset\\scaler.pkl'
        infile = open(fp,'rb')
        norm = pickle.load(infile)
        infile.close()
        X_test_norm = pd.DataFrame(norm.transform(X_test))
        X_test_norm.columns = X_test.columns
        X_test=X_test_norm.copy()

    pred_test=model.predict(X_test)
    return (pred_test)

def export_model(model, scale_flag, tune_key):
    import pickle
    pkl_file = os.getcwd()+'\\{}_{}_MODEL.pkl'.format(scale_flag, tune_key) # change this to your desired folder & file name
    data = model # define what you want to save here
    file = open(pkl_file,"wb")
    pickle.dump(data,file)
    file.close()

def import_model(fp):
    print('FP:', fp)
    import pickle
    infile = open(fp,'rb')
    model = pickle.load(infile)
    infile.close()
    return model
