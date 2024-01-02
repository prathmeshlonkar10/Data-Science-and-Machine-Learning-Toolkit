print('\n========== WELCOME ==========\n')

#import datetime
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import os
import PySimpleGUI as sg
import sys
sys.path.append(r'C:\Users\DELL\Desktop\Toolkit\Backend') #BACKEND FILE'S FOLDER PATH
import try_backend as tb # "try_backend" is BACKEND FILE

sg.theme('DefaultNoMoreNagging')
logo_path='C:\\Users\\DELL\\Desktop\\Toolkit\\intelimek.png'
sg.set_options(auto_size_buttons=False)
cwd=os.getcwd()
print('A: ', cwd)
dj_path, on_date_folder=tb.main_folder_creation(cwd)
print(dj_path)

# cwd = on_date_folder

track_dict  = tb.tracker_create()
inp_list    = tb.tracker_id_list()
EDA_1_list  = tb.tracker_id_list()
EDA_2_list  = tb.tracker_id_list()
model_list  = tb.tracker_id_list()
val_list    = tb.tracker_id_list()
tune_list   = tb.tracker_id_list()
exp_list    = tb.tracker_id_list()


tb.dump_tracker(track_dict)

# Definitions Start here...
def read_data(fp, hc):
    df, data, header_list, read_flag=tb.read_data(fp, hc) #BACKEND
    if read_flag==0:
        sg.popup_error('\nError reading file\n\n(Make sure a file is selected & its name is correct)\n', text_color='black', font='Helvetica')
        quit()
    else:
        return (df, data, header_list)

def which_set():
    if values['r1']==True:
        df, data, header_list, fn, r1=tb.which_set_train(train_df, train_header_list, fn1)
        #window['y_var'].update(values=header_list)
    elif values['r2']==True:
        df, data, header_list, fn, r1=tb.which_set_test(test_df, test_header_list, fn2)
    else:
        sg.popup_error('Select one of the datasets for analysis', font='Helvetica', text_color='blue')
        quit()
    return (df, data, header_list, fn, r1)


def data_show():
    df, data, header_list, fn, r1=which_set()
    layout = [
              [sg.Table(values=data,
                        headings=header_list,
                        alternating_row_color='grey',
                        vertical_scroll_only=False,
                        display_row_numbers=False,
                        pad=(30, 30),
                        font='Helvetica',
                        num_rows=min(25, len(data)),
                        auto_size_columns=True)]]
    window=sg.Window('Dataset window for {}'.format(fn), layout, size=(1250,550))
    event, values = window.read()
    window.close()

def head_show():
    df, data, header_list, fn, r1=which_set()
    hdata=tb.head_show(df)  #BACKEND = hdata=df.head().values.tolist()
    layout = [
              [sg.Table(values=hdata,
                        headings=header_list,
                        alternating_row_color='grey',
                        vertical_scroll_only=False,
                        pad=(30, 30),
                        font='Helvetica',
                        display_row_numbers=False,
                        num_rows=min(30, len(hdata)),
                        auto_size_columns=True)]]
    window=sg.Window('Head window for {}'.format(fn), layout, size=(1250,200))
    event, values = window.read()
    window.close()

def tail_show():
    df, data, header_list, fn, r1=which_set()
    tdata=tb.tail_show(df)  #BACKEND = tdata=df.tail().values.tolist()
    layout = [
              [sg.Table(values=tdata,
                        headings=header_list,
                        alternating_row_color='grey',
                        vertical_scroll_only=False,
                        pad=(30, 30),
                        font='Helvetica',
                        display_row_numbers=False,
                        num_rows=min(30, len(tdata)),
                        auto_size_columns=True)]]
    window=sg.Window('Tail window for {}'.format(fn), layout, size=(1250, 200))
    event, values = window.read()
    window.close()

def describe_show():
    df, data, header_list, fn, r1=which_set()
    ddata, header_list=tb.describe_show(df) #BACKEND
    layout = [
              [sg.Table(values=ddata,
                        headings=header_list,
                        alternating_row_color='grey',
                        vertical_scroll_only=False,
                        pad=(30, 30),
                        font='Helvetica',
                        display_row_numbers=False,
                        num_rows=min(25, len(ddata)),
                        auto_size_columns=True)]]
    window=sg.Window('Describe window for {}'.format(fn), layout)
    event, values = window.read()
    window.close()

def shape_show():
    df, data, header_list, fn, r1=which_set()
    shape, rows_with_nan=tb.shape_show(df) #BACKEND
    layout=[[sg.Text('\nShape of "{}" is:\n'.format(fn),
             font='Helvetica',
             text_color='blue')],
            [sg.Text('{}\n'.format(shape),
             font='Helvetica')],
            [sg.Text('Rows with NaN values are:\n',
             font='Helvetica',
             text_color='blue')],
            [sg.Text('{}\n'.format(rows_with_nan), size=(50, 6))]]
    window=sg.Window('Shape Window', layout, margins=(30,25))
    event, values=window.read()
    window.close()

def info_show():
    df, data, header_list, fn, r1=which_set()
    dtdata, header_list=tb.info_show(df)
    layout = [
              [sg.Table(values=dtdata,
                        headings=header_list,
                        alternating_row_color='grey',
                        vertical_scroll_only=False,
                        pad=(30, 30),
                        font='Helvetica',
                        display_row_numbers=False,
                        num_rows=min(25, len(dtdata)),
                        auto_size_columns=True)]]
    window=sg.Window('Information window', layout)
    event, values = window.read()
    window.close()


def drop_window():
    df, data, header_list, fn, r1=which_set()
    a=None
    updated_df=df.copy()
    updated_header=header_list.copy()
    layout = [
              [sg.Text('\nWhat do you wish to drop? ', font='Helvetica')],
              [sg.Text('Selected Dataset: {}'.format(fn), text_color='black')],
              [sg.Text('--'*49)],
              #[sg.Column(drop_row_col)],
              [sg.Text('ROWS\n', font='Helvetica', text_color='blue')],
              [sg.Text('Drop rows with?')],
              [sg.Radio('All NaN', 'row_del', default=False, key='rd1', tooltip=' Drops all rows with any NaN value ')],
              [sg.Radio('NaN in Column:\t', 'row_del', default=False, key='rd2', tooltip=' Drops all rows having NaN value in the specified column '),
               sg.Combo(list(updated_df.columns), enable_events=True, key='combo_for_row')],
              [sg.Radio('Threshold NaN:\t', 'row_del', default=False, key='rd3', tooltip=' Drops rows with NaN values greater than the specified threshold '),
               sg.Input(enable_events=True, key='thresh_inp', size=(17,1))],
              [sg.Radio('Index value:\t', 'row_del', default=False, key='rd4', tooltip=' Drops a row having the entered index value '),
               sg.Input(enable_events=True, key='index_inp', size=(17,1))],
              [sg.Button('Drop Row')],
              [sg.Text('', key='row_drop_update', text_color='black',size=(50,1))],
              #[sg.VSeperator()],
              [sg.Text('--'*49)],
              #[sg.Column(drop_col_col)],
              [sg.Text('COLUMNS\n', font='Helvetica', text_color='blue')],
              [sg.Text('Select the column you wish to drop:')],
              [sg.Combo(list(updated_df.columns), enable_events=True, key='combo_for_col')],
              [sg.Button('Drop Column')],
              [sg.Text('', key='col_drop_update', text_color='black',size=(40,1))],
              [sg.Text('')]
             ]
    window=sg.Window('Drop Selection Window', layout)
    while True:
        event, values = window.read()
        if event==sg.WIN_CLOSED:
            break
        if event=='Drop Row':
            a=r1
            if values['rd1']==True: # All NaN
                updated_df, updated_header, r1=tb.drop_rd1(r1, updated_df) #BACKEND
                window['row_drop_update'].update('Update: Rows with all NaN values dropped')

            if values['rd2']==True: # NaN in column
                col=values['combo_for_row']
                updated_df, updated_header, r1=tb.drop_rd2(r1, col, updated_df) #BACKEND
                window['row_drop_update'].update('Update: Rows with NaN in "{}" column dropped'.format(col))

            if values['rd3']==True: # Threshold NaN
                maxna=values['thresh_inp']
                updated_df, updated_header, r1=tb.drop_rd3(header_list, maxna, updated_df, r1) #BACKEND
                window['row_drop_update'].update('Update: Rows with more than {} NaN values dropped'.format(maxna))

            if values['rd4']==True:
                i=values['index_inp']
                updated_df, updated_header, r1=tb.drop_rd4(r1, i, updated_df) #BACKEND
                window['row_drop_update'].update('Update: Row with index value {} dropped (index reset)'.format(i))

        if event=='Drop Column':
            a=r1
            dcol=values['combo_for_col']
            updated_df, updated_header, r1=tb.drop_column(r1, dcol, updated_df) #BACKEND
            window['col_drop_update'].update('Update: "{}" column dropped'.format(dcol))
            window['combo_for_col'].update(values=list(updated_df.columns))
            window['combo_for_row'].update(values=list(updated_df.columns))

    window.close()
    return(updated_df, updated_header, a)

def replace_window():
    df, data, header_list, fn, r1=which_set()
    a=None
    num_cols, cat_cols =tb.num_cat_cols(df)

    num_col_selection_col=[
                           [sg.Text('Select column:', tooltip=' Select a column in which you wish to perform replace operation ')],
                           [sg.Radio('Greater than:', 'num_col_radio', default=False, key='ncr1', tooltip=' Replaces values greater than the specified value with the new value ')],
                           [sg.Radio('Lesser than:', 'num_col_radio', default=False, key='ncr2', tooltip=' Replaces values less than the specified value with the new value ')],
                           [sg.Radio('Unique value:', 'num_col_radio', default=False, key='ncr3', tooltip=' Replaces the particularly specified value with the new value ')],
                           [sg.Text('With New value:', tooltip=' The new value ')]
                          ]

    num_col_inputs_col=[
                        [sg.Combo(num_cols, enable_events=True, key='num_col_combo')],
                        [sg.Input(enable_events=True, key='ncr1_inp', size=(17,1))],
                        [sg.Input(enable_events=True, key='ncr2_inp', size=(17,1))],
                        [sg.Input(enable_events=True, key='ncr3_inp', size=(17,1))],
                        [sg.Input(enable_events=True, key='nv1_inp', size=(17,1))]
                       ]

    cat_col_selection_col=[
                           [sg.Text('Select column:  ', tooltip=' The column in which you wish to perform the replace operation ')],
                           [sg.Text('Unique value:  ', tooltip=' Replaces the particularly specified value with the new value ')],
                           [sg.Text('With New value:  ', tooltip=' The new value ')]
                          ]

    cat_col_inputs_col=[
                        [sg.Combo(cat_cols, enable_events=True, key='cat_col_combo')],
                        [sg.Input(enable_events=True, key='cc1_inp', size=(17,1))],
                        [sg.Input(enable_events=True, key='nv2_inp', size=(17,1))]
                       ]

    layout=[
            [sg.Text('\nSelect column to replace a value from', font='Helvetica')],
            [sg.Text('Selected Dataset: {}'.format(fn), text_color='black')],
            [sg.Text('--'*50)],
            [sg.Text('Replace from Numerical columns\n', font='Helvetica', text_color='blue')],
            [sg.Column(num_col_selection_col),
             sg.Column(num_col_inputs_col)],
            [sg.Button('Replace', key='replace_1')],
            [sg.Text('--'*50)],
            [sg.Text('Replace from Categorical columns\n', font='Helvetica', text_color='blue')],
            [sg.Column(cat_col_selection_col),
             sg.Column(cat_col_inputs_col)],
            [sg.Button('Replace', key='replace_2')],
            [sg.Text('--'*50)],
            [sg.Text('', key='update_replace', text_color='black', size=(45,2))],
            [sg.Text('')]
           ]
    window=sg.Window('Replace window', layout)
    while True:
        event, values=window.read()
        if event==sg.WIN_CLOSED:
            break
        if event=='replace_1': # For Numerical columns
            a=r1
            sel_num_col=values['num_col_combo']
            if values['ncr1']==True: # Greater than
                gt=values['ncr1_inp']
                nv=values['nv1_inp']
                tb.replace_ncr1(sel_num_col, gt, nv, df)
                window['update_replace'].update('Update: All values greater than {} in "{}" column replaced by {}'.format(gt, sel_num_col, nv))

            if values['ncr2']==True: # Lesser than
                lt=values['ncr2_inp']
                nv=values['nv1_inp']
                tb.replace_ncr2(sel_num_col, lt, nv, df)
                window['update_replace'].update('Update: All values lesser than {} in "{}" column replaced by {}'.format(lt, sel_num_col, nv))

            if values['ncr3']==True: # Unique value
                uv=values['ncr3_inp']
                nv=values['nv1_inp']
                tb.replace_ncr3(sel_num_col, uv, nv, df)
                window['update_replace'].update('Update: All unique values of {} in "{}" column replaced by {}'.format(uv, sel_num_col, nv))

        if event=='replace_2': # For Categorical columns
            a=r1
            sel_cat_col=values['cat_col_combo']
            ov=values['cc1_inp']
            nv=values['nv2_inp']
            tb.replace_cat(sel_cat_col, ov, nv, df)
            window['update_replace'].update('Update: All unique values of {} in "{}" column replaced by {}'.format(ov, sel_cat_col, nv))

    window.close()
    return(a)

def fillna_window():
    df, data, header_list, fn, r1=which_set()
    a=None
    num_cols, cat_cols =tb.num_cat_cols(df)
    num_cols.insert(0, 'All numerical columns')
    cat_cols.insert(0, 'All categorical columns')

    num_fill_sel=[
                  [sg.Text('Select Column:'),
                    sg.Combo(num_cols, enable_events=True, key='nfs_combo')]
                 ]

    num_fill_ops=[
                  [sg.Text('Select Method:')],
                  [sg.Radio('Mean', 'num_fill_ops', default=False, key='nfo1'),
                   sg.Radio('Median', 'num_fill_ops', default=False, key='nfo2')],
                  [sg.Radio('Mode', 'num_fill_ops', default=False, key='nfo3'),
                   sg.Radio('Interpolation', 'num_fill_ops', default=False, key='nfo4')],
                  [sg.Radio('Previous', 'num_fill_ops', default=False, key='nfo5')],
                  [sg.Radio('Unique:', 'num_fill_ops', default=False, key='nfo6'),
                   sg.Input(enable_events=True, key='nfo6_inp', size=(6,1))]
                 ]

    cat_fill_sel=[
                  [sg.Text('Select Column:'),
                   sg.Combo(cat_cols, enable_events=True, key='cfs_combo')]
                 ]

    cat_fill_ops=[
                  [sg.Text('Select Method:')],
                  [sg.Radio('Most frequent', 'cat_fill_ops', default=False, key='cfo1')],
                  [sg.Radio('Unique:', 'cat_fill_ops', default=False, key='cfo2'),
                   sg.Input(enable_events=True, key='cfo2_inp', size=(6,1))]
                 ]

    layout=[
            [sg.Text('\nFill NaN values in', font='Helvetica')],
            [sg.Text('Selected Dataset is: {}'.format(fn), text_color='black')],
            [sg.Text('--'*60)],
            [sg.Text('NUMERICAL COLUMNS\n', font='Helvetica', text_color='blue')],
            [sg.Column(num_fill_sel),
             sg.VSeperator(),
             sg.Column(num_fill_ops)],
            [sg.Button('Fill', key='num_fill')],
            [sg.Text('', key='num_fill_update', text_color='black', size=(40,1))],
            [sg.Text('--'*60)],
            [sg.Text('CATEGORICAL COLUMNS\n', font='Helvetica', text_color='blue')],
            [sg.Column(cat_fill_sel),
             sg.VSeperator(),
             sg.Column(cat_fill_ops)],
            [sg.Button('Fill', key='cat_fill')],
            [sg.Text('', key='cat_fill_update', text_color='black', size=(40,1))],
            [sg.Text('')]
           ]
    window=sg.Window('Fill NaN Window', layout)
    while True:
        event,values=window.read()
        if event==sg.WIN_CLOSED:
            break
        if event=='num_fill':
            a=r1
            sel_num_col=values['nfs_combo']
            if values['nfo1']==True: # Mean
                tb.fill_num_mean(df, sel_num_col)
                window['num_fill_update'].update('NaN values in "{}" filled with Mean'.format(sel_num_col))

            if values['nfo2']==True: # Median
                tb.fill_num_median(df, sel_num_col)
                window['num_fill_update'].update('NaN values in "{}" filled with Median'.format(sel_num_col))

            if values['nfo3']==True: # Mode
                tb.fill_num_mode(df, sel_num_col)
                window['num_fill_update'].update('NaN values in "{}" filled with Mode'.format(sel_num_col))

            if values['nfo4']==True: # Interpolation
                tb.fill_num_interpolation(df, sel_num_col)
                window['num_fill_update'].update('NaN values in "{}" filled with Interpolation'.format(sel_num_col))

            if values['nfo5']==True: # Previous
                tb.fill_num_previous(df, sel_num_col)
                window['num_fill_update'].update('NaN values in "{}" filled with Preceding values'.format(sel_num_col))

            if values['nfo6']==True: # Unique
                uv=values['nfo6_inp']
                tb.fill_num_unique(df, uv, sel_num_col)
                window['num_fill_update'].update('NaN values in "{}" filled with "{}"'.format(sel_num_col, uv))

        if event=='cat_fill':
            a=r1
            sel_cat_col=values['cfs_combo']
            if values['cfo1']==True: # Most frequent
                tb.fill_cat_mf(df, sel_cat_col)
                window['cat_fill_update'].update('NaN values in "{}" filled with Most frequent value'.format(sel_cat_col))

            if values['cfo2']==True: # Unique
                uv=values['cfo2_inp']
                tb.fill_cat_uv(df, uv, sel_cat_col)
                window['cat_fill_update'].update('NaN values in "{}" filled with "{}"'.format(sel_cat_col, uv))

    window.close()
    return(a)

def convert_window():
    df, data, header_list, fn, r1=which_set()
    a=None
    dtype_list=tb.dtype_list()
    layout=[
            [sg.Text('')],
            [sg.Text('Select a column:\t'),
             sg.Combo(header_list, enable_events=True, key='combo_convert_col')],
            [sg.Text('Convert into dtype:\t'),
             sg.Combo(dtype_list, enable_events=True, key='combo_convert_into')],
            [sg.Text('')],
            [sg.Button('OK')],
            [sg.Text('', text_color='black', key='dtype_convert_update', size=(45,2))],
            [sg.Text('')]
           ]
    window=sg.Window('Dtype conversion window', layout)
    while True:
        event, values=window.read()
        if event==sg.WIN_CLOSED:
            break
        if event=='OK':
            a=r1
            col=values['combo_convert_col']
            dtype=values['combo_convert_into']
            cdtf=tb.convert_dtype(df, col, dtype)
            if cdtf==1:
                sg.popup_error('\nError\n\nEnsure that the "{}" coulmn has no NaN values\n'.format(col), font='Helvetica', text_color='black')
            window['dtype_convert_update'].update('Dtype of "{}" column converted to "{}"'.format(col, dtype))
    window.close()
    return (a)

def unique_sel_window(): # DISCONTINUED
    df, data, header_list, fn, r1=which_set()
    num_cols, cat_cols =tb.num_cat_cols(df)
    layout=[
            [sg.Text('')],
            [sg.Text('Select a column:'),
             sg.Combo(cat_cols, enable_events=True, key='combo_unique')],
            [sg.Text('')],
            [sg.Button('OK')]
           ]
    window=sg.Window('Column selection for unique', layout)
    while True:
        event, values=window.read()
        if event==sg.WIN_CLOSED:
            break
        if event=='OK':
            col=values['combo_unique']
            unique=tb.unique_sel_window(df, col)
            layout2=[
                     [sg.Text('\nUnique values in "{}" column are:\n'.format(col), font='Helvetica', text_color='blue')],
                     [sg.Text('{}\n'.format(unique.tolist()), font='Helvetica')]
                    ]
            window2=sg.Window('Unique values display', layout2, margins=(50,25))
            event, values=window2.read()
            window2.close()
    window.close()

def value_counts_window():
    df, data, header_list, fn, r1=which_set()
    num_cols, cat_cols =tb.num_cat_cols(df)
    layout=[
            [sg.Text('')],
            [sg.Text('Select a column:'),
             sg.Combo(cat_cols, enable_events=True, key='combo_value_count')],
            [sg.Text('')],
            [sg.Button('OK')]
           ]
    window=sg.Window('Column selection for value count', layout)
    while True:
        event, values=window.read()
        if event==sg.WIN_CLOSED:
            break
        if event=='OK':
            col=values['combo_value_count']
            vcdt, vchl=tb.value_counts_window(df, col)
            layout2=[
                     [sg.Text('\nNote: NaN values counts(if any) have been excluded\n', text_color='blue', font='Helvetica')],
                     [sg.Table(values=vcdt,
                               headings=vchl,
                               alternating_row_color='grey',
                               vertical_scroll_only=False,
                               pad=(30, 30),
                               font='Helvetica',
                               display_row_numbers=False,
                               num_rows=min(25, len(vcdt)),
                               auto_size_columns=True)]]
            window2=sg.Window('Value Count for column "{}"'.format(col), layout2)
            event, values=window2.read()
            window2.close()
    window.close()

def label_encode_window():
    df, data, header_list, fn, r1=which_set()
    a=None
    num_cols, cat_cols =tb.num_cat_cols(df)
    cat_cols.insert(0, 'All categorical columns')
    layout = [
              [sg.Text('')],
              [sg.Text('Select a column:\t'),
               sg.Combo(cat_cols, enable_events=True, key='combo_encode_col')],
              #[sg.Radio('Normalize', 'scaling', default=False, key='scale_norm')],
              #[sg.Radio('Standardize', 'scaling', default=False, key='scale_stand')],
              [sg.Text('')],
              [sg.Button('OK')],
              [sg.Text('', text_color='black', key='Encode_update', size=(45,2))],
              [sg.Text('')]
             ]
    window=sg.Window('Label Encode window', layout)
    while True:
        event, values=window.read()
        if event==sg.WIN_CLOSED:
            break
        if event=='OK':
            a=r1
            col=values['combo_encode_col']
            tb.encode_window(df, col)
            window['Encode_update'].update('Label encoding for "{}" column done'.format(col))

    window.close()
    return (a)


def import_window():
    layout=[
             [sg.Text('')],
             [sg.Text('Model:\t', font='Helvetica'),
              sg.In(enable_events=True, key='inp_pkl'),
              sg.FileBrowse(file_types=(('PKL Files', '*.pkl'),))],
             [sg.Text('')],
             [sg.Button('OK')],
             [sg.Text('', text_color='black', size=(30,1), key='update_import')],
             [sg.Text('')]
        ]
    window=sg.Window('Model import window', layout)
    while True:
        event, values=window.read()
        if event==sg.WIN_CLOSED:
            break
        if event=='OK':
            fp=values['inp_pkl']
            model=tb.import_model(fp)
            window['update_import'].update('Update: Model imported successfully!')
            return(fp, model)
    window.close()

def model_window(df):
    # df taken as argument
    # df, data, header_list, fn, r1=which_set()
    algorithms = tb.algo_list() #BACKEND
    inp_dict=tb.tracker_id_dict()
    layout=[
            [sg.Text('\nFollow the steps to correctly select & apply a model\t\n', font='Helvetica', text_color='blue')],
            [sg.Text('1) Select target variable:\t'),
             sg.Combo(list(df.columns), enable_events=True, key='y_var')],
            [sg.Text('2) Train-validation split:\t'),
             sg.Radio('Yes', 'tvs', default=False, key='tvs_yes'),
             sg.Radio('No', 'tvs', default=False, key='tvs_no')],
            [sg.Text('3) Normal Scaling:\t\t'),
             sg.Radio('Yes', 'scale_norm', default=False, key='norm_yes'),
             sg.Radio('No', 'scale_norm', default=False, key='norm_no')],
            [sg.Text('4) Standard Scaling:\t'),
             sg.Radio('Yes', 'scale_stand', default=False, key='stand_yes'),
             sg.Radio('No', 'scale_stand', default=False, key='stand_no')],
            [sg.Text('5) Select model algorithm:\t'),
             sg.Combo(algorithms, enable_events=True, key='algo')],
            [sg.Text('')],
            [sg.Button('OK')],
            [sg.Text('', key='fit_update', size=(30,1), text_color='black')],
            [sg.Text('')]
           ]

    window=sg.Window('Model Selection Window', layout)
    while True:
        event, values=window.read()
        if event==sg.WIN_CLOSED:
            break
        if event=='OK':
            print('POP')
            inp_dict=tb.tracker_dict_mod(inp_dict, 'run_id', tc)
            print(tc)
            target_col=values['y_var']
            X, y = tb.target_selection(df, target_col) #BACKEND
            inp_dict=tb.tracker_dict_mod(inp_dict, 'target_variable', target_col)
            #tb.tracker_add_list(model_list, 'model', 'target_variable : {}'.format(target_col))

            if values['tvs_yes']==True:
                X_train, X_val, y_train, y_val=tb.TVS_yes(X, y) #BACKEND
                tvs_flag=1
                inp_dict=tb.tracker_dict_mod(inp_dict, 'train_validation_split', 'yes')
                #tb.tracker_add_list(model_list, 'model', 'train_validation_split : Yes')
            else:
                X_train, X_val, y_train, y_val=tb.TVS_no(X, y) #BACKEND
                tvs_flag=0
                inp_dict=tb.tracker_dict_mod(inp_dict, 'train_validation_split', 'no')
                #tb.tracker_add_list(model_list, 'model', 'train_validation_split : No')

            if values['norm_yes']==True:
                X_train, X_val, scale_flag = tb.normalize(tvs_flag, X_train, X_val)
                if scale_flag==None:
                    sg.popup_error('\nERROR: Could not Normalize the model.\n\nTip: Ensure that the data is cleaned or transformed properly.\n\nTry again !\n')
                    continue
                inp_dict=tb.tracker_dict_mod(inp_dict, 'normalization', 'yes')
                #tb.tracker_add_list(model_list, 'model', 'normalization : Yes')
            else:
                scale_flag=0
                inp_dict=tb.tracker_dict_mod(inp_dict, 'normalization', 'no')
                #tb.tracker_add_list(model_list, 'model', 'normalization : No')

            if values['stand_yes']==True:
                X_train, X_val, scale_flag = tb.standardize(tvs_flag, X_train, X_val)
                if scale_flag==None:
                    sg.popup_error('\nERROR: Could not Standardize the model.\n\nTip: Ensure that the data is cleaned or transformed properly.\n\nTry again !\n')
                    continue
                inp_dict=tb.tracker_dict_mod(inp_dict, 'standardization', 'yes')
                #tb.tracker_add_list(model_list, 'model', 'standardization : Yes')
            else:
                inp_dict=tb.tracker_dict_mod(inp_dict, 'standardization', 'no')
                #tb.tracker_add_list(model_list, 'model', 'standardization : No')
                pass

            algo=values['algo']
            print(algo, type(algo)) # cross-check
            inp_dict=tb.tracker_dict_mod(inp_dict, 'algorithm', algo)
            #tb.tracker_add_list(model_list, 'model', 'algorithm : {}'.format(algo))
            model, tune_key=tb.fit_model(algo, X_train, y_train) #BACKEND
            print(model) # cross-check
            if model==None:
                #tb.tracker_add_list(model_list, 'model', 'fit : Failed')
                sg.popup_error('\nERROR: Could not fit the model.\n\nTip: Ensure that the data is cleaned or transformed properly.\n\nTry again !\n')
                continue
            else:
                window['fit_update'].update('Update: Model fitting successful !')
                #inp_dict=tb.tracker_dict_mod(inp_dict, 'tune', 'no')
                #inp_dict=tb.tracker_dict_mod(inp_dict, 'validate', 'no')
                #inp_dict=tb.tracker_dict_mod(inp_dict, 'export', 'no')
                tb.tracker_add_list(model_list, 'model', inp_dict)
                #tb.tracker_add_status('no')
                return (model, algo, tvs_flag, scale_flag, tune_key, X_train, X_val, y_train, y_val)
    window.close()

def tune_window(model, algo, tune_key, X_train, y_train):
    print('Tuning')
    if tune_key==1:  # LOGISTIC REGRESSION
        penalty=tb.lor_penalty()
        solver=tb.lor_solver()
        layout=[
                [sg.Text('\nParameters to tune: (Enter in every field)\n', font='Helvetica', text_color='blue')],
                [sg.Text('1) penalty:\t'), # selection
                 sg.Combo(penalty, enable_events=True, key='penalty_combo')],
                [sg.Text('2) C:\t\t'), # float
                 sg.Input('1',enable_events=True, key='C_inp', size=(15,1))],
                [sg.Text('3) solver:\t\t'), # selection
                 sg.Combo(solver, enable_events=True, key='solver_combo')],
                [sg.Text('')],
                [sg.Button('OK')],
                [sg.Text('', text_color='black', key='para_update', size=(30,2))],
                [sg.Text('')]]
        window=sg.Window('Tune window for Logistic Regression', layout)
        while True:
            print('RS')
            event, values=window.read()
            if event==sg.WIN_CLOSED:
                break
            if event=='OK':
                penalty_val=values['penalty_combo']
                C_val=float(values['C_inp'])
                solver_val=values['solver_combo']
                model, fit_flag=tb.tune_lor(penalty_val, C_val, solver_val, X_train, y_train)
                if fit_flag==1:
                    window['para_update'].update('Update: Parameters applied')
                else:
                    window['para_update'].update('Error: Try again with correct set of parameters!')
                    continue
                return(model)
        window.close()
    if tune_key==2:  # KNN CLASSIFIER
        layout=[
                [sg.Text('')],
                [sg.Text('1) n_neighbors: ', text_color='blue', font='Helvetica'),
                 # int
                 sg.Input('5',enable_events=True, key='nn_inp', size=(8,1))],
                [sg.Text('')],
                [sg.Button('OK')],
                [sg.Text('', text_color='black', key='para_update', size=(30,2))],
                [sg.Text('')]
               ]
        window=sg.Window('Tune window for KNN Classifier', layout)
        while True:
            event, values=window.read()
            if event==sg.WIN_CLOSED:
                break
            if event=='OK':
                nn=int(values['nn_inp'])
                model, fit_flag=tb.tune_knn_c(nn, X_train, y_train)
                if fit_flag==1:
                    window['para_update'].update('Update: Parameters applied')
                else:
                    window['para_update'].update('Error: Try again with correct parameter!')
                    continue
                return(model)
        window.close()
    if tune_key==3:  # SVM
        kernel=tb.svm_kernel()
        layout=[
                [sg.Text('\nParameters to tune: (Enter in every field)\n', font='Helvetica', text_color='blue')],
                [sg.Text('1) kernel:\t\t'), # selection
                 sg.Combo(kernel, enable_events=True, key='kernel_combo')],
                [sg.Text('2) C:\t\t'), # float
                 sg.Input('1', enable_events=True, key='C_inp', size=(10,1))],
                [sg.Text('3) degree:\t'), # int
                 sg.Input('3', enable_events=True, key='degree_inp', size=(10,1))],
                [sg.Text('4) gamma:\t'), # float
                 sg.Input('scale', enable_events=True, key='gamma_inp', size=(10,1))],
                [sg.Text('')],
                [sg.Button('OK')],
                [sg.Text('', text_color='black', key='para_update', size=(30,2))],
                [sg.Text('')]
               ]

        window=sg.Window('Tune window for SVM', layout)
        while True:
            event, values = window.read()
            if event==sg.WIN_CLOSED:
                break
            if event=='OK':
                kernel_val=values['kernel_combo']
                C_val=float(values['C_inp'])
                degree_val=int(values['degree_inp'])
                gamma_val=values['gamma_inp']
                if gamma_val!='scale':
                    gamma_val=float(values['gamma_inp'])
                model, fit_flag=tb.tune_svm(kernel_val, C_val, degree_val, gamma_val ,X_train, y_train)
                if fit_flag==1:
                    window['para_update'].update('Update: Parameters applied')
                else:
                    window['para_update'].update('Error: Try again with correct set of parameters!')
                    continue
                return(model)
        window.close()
    if tune_key==4:  # DECISION TREE CLASSIFIER
        layout=[
                [sg.Text('\nParameters to tune: (Enter in every field)\n', font='Helvetica', text_color='blue')],
                [sg.Text('1) min_samples_split:\t'), # int
                 sg.Input('2', enable_events=True, key='mss_inp', size=(10,1))],
                [sg.Text('2) min_samples_leaf:\t'), # int
                 sg.Input('1', enable_events=True, key='msl_inp', size=(10,1))],
                [sg.Text('3) max_depth:\t\t'), # int
                 sg.Input('None', enable_events=True, key='md_inp', size=(10,1))],
                [sg.Text('4) max_leaf_nodes:\t'), # int
                 sg.Input('None', enable_events=True, key='mln_inp', size=(10,1))],
                [sg.Text('')],
                [sg.Button('OK')],
                [sg.Text('', text_color='black', key='para_update', size=(30,2))],
                [sg.Text('')]
               ]

        window=sg.Window('Tune window for Decision-Tree Classifier', layout)
        while True:
            event, values = window.read()
            if event==sg.WIN_CLOSED:
                break
            if event=='OK':
                mss_val=int(values['mss_inp'])
                msl_val=int(values['msl_inp'])

                md_val=values['md_inp']
                if md_val=='None':
                    md_val=None
                else:
                    md_val=int(values['md_inp'])

                mln_val=values['mln_inp']
                if mln_val=='None':
                    mln_val=None
                else:
                    mln_val=int(values['mln_inp'])
                model, fit_flag=tb.tune_dtc(mss_val, msl_val, md_val, mln_val ,X_train, y_train)
                if fit_flag==1:
                    window['para_update'].update('Update: Parameters applied')
                else:
                    window['para_update'].update('Error: Try again with correct set of parameters!')
                    continue
                return(model)
        window.close()
    if tune_key==5:  # RANDOM FOREST CLASSIFIER
        layout=[
                [sg.Text('\nParameters to tune: (Enter in every field)\n', font='Helvetica', text_color='blue')],
                [sg.Text('1) n_estimators:\t'), # int
                 sg.Input('100', enable_events=True, key='ne_inp', size=(10,1))],
                [sg.Text('2) min_samples_split:\t'), # int
                 sg.Input('2', enable_events=True, key='mss_inp', size=(10,1))],
                [sg.Text('3) min_samples_leaf:\t'), # int
                 sg.Input('1', enable_events=True, key='msl_inp', size=(10,1))],
                [sg.Text('4) max_depth:\t\t'), # int
                 sg.Input('None', enable_events=True, key='md_inp', size=(10,1))],
                [sg.Text('5) max_leaf_nodes:\t'), # int
                 sg.Input('None', enable_events=True, key='mln_inp', size=(10,1))],
                [sg.Text('')],
                [sg.Button('OK')],
                [sg.Text('', text_color='black', key='para_update', size=(30,2))],
                [sg.Text('')]
               ]

        window=sg.Window('Tune window for Random-Forest Classifier', layout)
        while True:
            event, values = window.read()
            if event==sg.WIN_CLOSED:
                break
            if event=='OK':
                ne_val=int(values['ne_inp'])
                mss_val=int(values['mss_inp'])
                msl_val=int(values['msl_inp'])
                md_val=values['md_inp']
                if md_val=='None':
                    md_val=None
                else:
                    md_val=int(values['md_inp'])

                mln_val=values['mln_inp']
                if mln_val=='None':
                    mln_val=None
                else:
                    mln_val=int(values['mln_inp'])
                model, fit_flag=tb.tune_rfc(ne_val, mss_val, msl_val, md_val, mln_val ,X_train, y_train)
                if fit_flag==1:
                    window['para_update'].update('Update: Parameters applied')
                else:
                    window['para_update'].update('Error: Try again with correct set of parameters!')
                    continue
                return(model)
        window.close()

    if tune_key==6:  # LINEAR REGRESSION
        layout=[sg.Text('\nNote: No parameters to tune for this algorithm\n', text_color='blue', font='Helvetica', size=(20, 2))]
        window=sg.Window('Tune window for Linear Regression', layout)
        event, values=window.read()
        window.close()
    if tune_key==7:  # KNN REGRESSOR
        layout=[
                [sg.Text('')],
                [sg.Text('1) n_neighbors: ', text_color='blue', font='Helvetica'),
                 # int
                 sg.Input('5',enable_events=True, key='nn_inp', size=(8,1))],
                [sg.Text('')],
                [sg.Button('OK')],
                [sg.Text('', text_color='black', key='para_update', size=(30,2))],
                [sg.Text('')]
               ]
        window=sg.Window('Tune window for KNN Regressor', layout)
        while True:
            event, values=window.read()
            if event==sg.WIN_CLOSED:
                break
            if event=='OK':
                nn=int(values['nn_inp'])
                model, fit_flag=tb.tune_knn_r(nn, X_train, y_train)
                if fit_flag==1:
                    window['para_update'].update('Update: Parameters applied')
                else:
                    window['para_update'].update('Error: Try again with correct parameter!')
                    continue
                return(model)
        window.close()
    if tune_key==8:  # SVR
        kernel=tb.svm_kernel()
        layout=[
                [sg.Text('\nParameters to tune: (Enter in every field)\n', font='Helvetica', text_color='blue')],
                [sg.Text('1) kernel:\t\t'), # selection
                 sg.Combo(kernel, enable_events=True, key='kernel_combo')],
                [sg.Text('2) epsilon:\t\t'), # float
                 sg.Input('0.1', enable_events=True, key='epsilon_inp', size=(10,1))],
                [sg.Text('3) C:\t\t'), # float
                 sg.Input('1', enable_events=True, key='C_inp', size=(10,1))],
                [sg.Text('4) degree:\t'), # int
                 sg.Input('3', enable_events=True, key='degree_inp', size=(10,1))],
                [sg.Text('5) gamma:\t'), # float
                 sg.Input('scale', enable_events=True, key='gamma_inp', size=(10,1))],
                [sg.Text('')],
                [sg.Button('OK')],
                [sg.Text('', text_color='black', key='para_update', size=(30,2))],
                [sg.Text('')]
               ]

        window=sg.Window('Tune window for SVR', layout)
        while True:
            event, values = window.read()
            if event==sg.WIN_CLOSED:
                break
            if event=='OK':
                kernel_val=values['kernel_combo']
                epsilon_val=float(values['epsilon_inp'])
                C_val=float(values['C_inp'])
                degree_val=int(values['degree_inp'])
                gamma_val=values['gamma_inp']
                if gamma_val!='scale':
                    gamma_val=float(values['gamma_inp'])
                model, fit_flag=tb.tune_svr(kernel_val, epsilon_val, C_val, degree_val, gamma_val ,X_train, y_train)
                if fit_flag==1:
                    window['para_update'].update('Update: Parameters applied')
                else:
                    window['para_update'].update('Error: Try again with correct set of parameters!')
                    continue
                return(model)
        window.close()
    if tune_key==9:  # DECISION TREE REGRESSOR (same as DTC)
        layout=[
                [sg.Text('\nParameters to tune: (Enter in every field)\n', font='Helvetica', text_color='blue')],
                [sg.Text('1) min_samples_split:\t'), # int
                 sg.Input('2', enable_events=True, key='mss_inp', size=(10,1))],
                [sg.Text('2) min_samples_leaf:\t'), # int
                 sg.Input('1', enable_events=True, key='msl_inp', size=(10,1))],
                [sg.Text('3) max_depth:\t\t'), # int
                 sg.Input('None', enable_events=True, key='md_inp', size=(10,1))],
                [sg.Text('4) max_leaf_nodes:\t'), # int
                 sg.Input('None', enable_events=True, key='mln_inp', size=(10,1))],
                [sg.Text('')],
                [sg.Button('OK')],
                [sg.Text('', text_color='black', key='para_update', size=(30,2))],
                [sg.Text('')]
               ]

        window=sg.Window('Tune window for Decision-Tree Regressor', layout)
        while True:
            event, values = window.read()
            if event==sg.WIN_CLOSED:
                break
            if event=='OK':
                mss_val=int(values['mss_inp'])
                msl_val=int(values['msl_inp'])
                md_val=values['md_inp']
                if md_val=='None':
                    md_val=None
                else:
                    md_val=int(values['md_inp'])

                mln_val=values['mln_inp']
                if mln_val=='None':
                    mln_val=None
                else:
                    mln_val=int(values['mln_inp'])
                model, fit_flag=tb.tune_dtr(mss_val, msl_val, md_val, mln_val ,X_train, y_train)
                if fit_flag==1:
                    window['para_update'].update('Update: Parameters applied')
                else:
                    window['para_update'].update('Error: Try again with correct set of parameters!')
                    continue
                return(model)
        window.close()
    if tune_key==10: # RANDOM FOREST REGRESSOR (same as RFC)
        layout=[
                [sg.Text('\nParameters to tune: (Enter in every field)\n', font='Helvetica', text_color='blue')],
                [sg.Text('1) n_estimators:\t'),
                 sg.Input('100', enable_events=True, key='ne_inp', size=(10,1))],
                [sg.Text('2) min_samples_split:\t'), # int
                 sg.Input('2', enable_events=True, key='mss_inp', size=(10,1))],
                [sg.Text('3) min_samples_leaf:\t'), # int
                 sg.Input('1', enable_events=True, key='msl_inp', size=(10,1))],
                [sg.Text('4) max_depth:\t\t'), # int
                 sg.Input('None', enable_events=True, key='md_inp', size=(10,1))],
                [sg.Text('5) max_leaf_nodes:\t'), # int
                 sg.Input('None', enable_events=True, key='mln_inp', size=(10,1))],
                [sg.Text('')],
                [sg.Button('OK')],
                [sg.Text('', text_color='black', key='para_update', size=(30,2))],
                [sg.Text('')]
               ]

        window=sg.Window('Tune window for Random-Forest Regressor', layout)
        while True:
            event, values = window.read()
            if event==sg.WIN_CLOSED:
                break
            if event=='OK':
                ne_val=int(values['ne_inp'])
                mss_val=int(values['mss_inp'])
                msl_val=int(values['msl_inp'])
                md_val=values['md_inp']
                if md_val=='None':
                    md_val=None
                else:
                    md_val=int(values['md_inp'])

                mln_val=values['mln_inp']
                if mln_val=='None':
                    mln_val=None
                else:
                    mln_val=int(values['mln_inp'])
                model, fit_flag=tb.tune_rfr(ne_val, mss_val, msl_val, md_val, mln_val ,X_train, y_train)
                if fit_flag==1:
                    window['para_update'].update('Update: Parameters applied')
                else:
                    window['para_update'].update('Error: Try again with correct set of parameters!')
                    continue
                return(model)
        window.close()

def validate_fxn(model, tvs_flag, tune_key, X_val, X_train, dvfSVR):
    pred_val, pred_train, pvf=tb.validation_pred(model, tvs_flag, tune_key, X_val, X_train, dvfSVR)
    if pvf==0:
        #sg.popup_error('Error: Validation set was not initiated by the user while\nselecting the model ')
        layout=[
                [sg.Text('\nPrediction for only training set completed !\n\nExit this window to continue\n', text_color='blue', font='Helvetica')]
               ]
        window=sg.Window('Validation update window', layout, margins=(30,25))
        event, values=window.read()
        window.close()
    else:
        layout=[
                [sg.Text('\nPrediction for training & validation set completed !\n\nExit this window to continue\n', text_color='blue', font='Helvetica')]
               ]
        window=sg.Window('Validation update window', layout, margins=(30,25))
        event, values=window.read()
        window.close()
    return (pred_val, pred_train, pvf)

def metric_selection_window():
    classification_metrics=tb.classification_metrics()
    regression_metrics=tb.regression_metrics()
    layout=[
            [sg.Text('\nSelect a metric for evaluation\n', font='Helvetica')],
            [sg.Text('--'*30)],
            [sg.Text('Classification metrics:', font='Helvetica', text_color='blue')],
            [sg.Combo(classification_metrics, enable_events=True, key='cm_combo')],
            [sg.Button('OK', key='OK_cm')],
            [sg.Text('--'*30)],
            [sg.Text('Regression metrics:', font='Helvetica', text_color='blue')],
            [sg.Combo(regression_metrics, enable_events=True, key='rm_combo')],
            [sg.Button('OK', key='OK_rm')],
            [sg.Text('--'*30)],
            [sg.Text('', text_color='black', key='metrics_update', size=(30,2))],
            [sg.Text('')]
           ]
    window=sg.Window('Metric selection window', layout)
    while True:
        event, values=window.read()
        if event==sg.WIN_CLOSED:
            break
        if event=='OK_cm':
            metric=values['cm_combo']
            window['metrics_update'].update('Update: {} metric selected for classification'. format(metric))
            return metric
        if event=='OK_rm':
            metric=values['rm_combo']
            window['metrics_update'].update('Update: {} metric selected for regression'. format(metric))
            return metric
    window.close()

def metric_display(result_train, result_val, mf):
    if mf=='c1': # Confusion Matrix FORMAT DISCONTINUED
        data=result.tolist()
        header_list=['class no.' + str(x+1) for x in range(len(data))]
        layout = [
                  [sg.Text('\nConfusion Matrix:', font='Helvetica', text_color='blue')],
                  [sg.Text('1) Rows represent actual values')],
                  [sg.Text('2) Columns represent predicted values')],
                  [sg.Table(values=data,
                            headings=header_list,
                            alternating_row_color='grey',
                            vertical_scroll_only=True,
                            pad=(20, 20),
                            font='Helvetica',
                            display_row_numbers=False,
                            num_rows=min(30, len(data)),
                            auto_size_columns=True)]]
        window=sg.Window('Confusion Matrix', layout)
        event, values=window.read()
        window.close()
    if mf=='c2': # Accuracy_score
        layout=[
                [sg.Text('\nAccuracy Score is:\n', font='Helvetica', text_color='blue')],
                [sg.Text('Training accuracy:\t{}'.format(result_train), font='Helvetica')],
                [sg.Text('Validation accuracy:\t{}\n'.format(result_val), font='Helvetica')]
               ]
        window=sg.Window('Accuracy metric window', layout, margins=(45,30))
        event, values=window.read()
        window.close()
    if mf=='c3': # Precision & Recall
        prdata, header_list=tb.pr_df(result_train, result_val)
        layout = [
                  [sg.Table(values=prdata,
                            headings=header_list,
                            alternating_row_color='grey',
                            vertical_scroll_only=False,
                            pad=(35, 35),
                            font='Helvetica',
                            display_row_numbers=False,
                            num_rows=min(25, len(prdata)),
                            auto_size_columns=True)]]
        window=sg.Window('Precision-Recall window', layout)
        event, values = window.read()
        window.close()
    if mf=='c4': # F1-score
        layout=[
                [sg.Text('\nF1-Score is:\n', font='Helvetica', text_color='blue')],
                [sg.Text('For Training set:\t{}'.format(result_train), font='Helvetica')],
                [sg.Text('For Validation set:\t{}\n'.format(result_val), font='Helvetica')]
               ]
        window=sg.Window('F1-score window', layout, margins=(45,30))
        event, values=window.read()
        window.close()
    if mf=='c5': # AUC-ROC
        layout=[
                [sg.Text('\nAUC-ROC-Score is:\n', font='Helvetica', text_color='blue')],
                [sg.Text('For Training set:\t{}'.format(result_train), font='Helvetica')],
                [sg.Text('For Validation set:\t{}\n'.format(result_val), font='Helvetica')]
               ]
        window=sg.Window('AUC-ROC-score window', layout, margins=(45,30))
        event, values=window.read()
        window.close()
    if mf=='c6': # Log-Loss
        layout=[
                [sg.Text('\nLog-loss Score is:\n', font='Helvetica', text_color='blue')],
                [sg.Text('For Training set:\t{}'.format(result_train), font='Helvetica')],
                [sg.Text('For Validation set:\t{}\n'.format(result_val), font='Helvetica')]
               ]
        window=sg.Window('Log-loss score window', layout, margins=(45,30))
        event, values=window.read()
        window.close()

    if mf=='r1': # MAE
        layout=[
                [sg.Text('\nMean Absolute Error is:\n', font='Helvetica', text_color='blue')],
                [sg.Text('For Training set:\t{}'.format(result_train), font='Helvetica')],
                [sg.Text('For Validation set:\t{}\n'.format(result_val), font='Helvetica')]
               ]
        window=sg.Window('MAE window', layout, margins=(45,30))
        event, values=window.read()
        window.close()
    if mf=='r2': # MSE
        layout=[
                [sg.Text('\nMean Squared Error is:\n', font='Helvetica', text_color='blue')],
                [sg.Text('For Training set:\t{}'.format(result_train), font='Helvetica')],
                [sg.Text('For Validation set:\t{}\n'.format(result_val), font='Helvetica')]
               ]
        window=sg.Window('MSE window', layout, margins=(45,30))
        event, values=window.read()
        window.close()
    if mf=='r3': # RMSE
        layout=[
                [sg.Text('\n Root Mean Squared Error is:\n', font='Helvetica', text_color='blue')],
                [sg.Text('For Training set:\t{}'.format(result_train), font='Helvetica')],
                [sg.Text('For Validation set:\t{}\n'.format(result_val), font='Helvetica')]
               ]
        window=sg.Window('RMSE window', layout, margins=(45,30))
        event, values=window.read()
        window.close()
    if mf=='r4': # RMSLE
        layout=[
                [sg.Text('\n Root Mean Squared Log Error is:\n', font='Helvetica', text_color='blue')],
                [sg.Text('For Training set:\t{}'.format(result_train), font='Helvetica')],
                [sg.Text('For Validation set:\t{}\n'.format(result_val), font='Helvetica')]
               ]
        window=sg.Window('RMSLE window', layout, margins=(45,30))
        event, values=window.read()
        window.close()
    if mf=='r5': # R2 SCORE
        layout=[
                [sg.Text('\n R2 Score is:\n', font='Helvetica', text_color='blue')],
                [sg.Text('For Training set:\t{}'.format(result_train), font='Helvetica')],
                [sg.Text('For Validation set:\t{}\n'.format(result_val), font='Helvetica')]
               ]
        window=sg.Window('R2 window', layout, margins=(45,30))
        event, values=window.read()
        window.close()
    if mf=='r6': # ADJUSTED R2 SCORE
        layout=[
                [sg.Text('\n Adjusted R2 Score is:\n', font='Helvetica', text_color='blue')],
                [sg.Text('For Training set:\t{}'.format(result_train), font='Helvetica')],
                [sg.Text('For Validation set:\t{}\n'.format(result_val), font='Helvetica')]
               ]
        window=sg.Window('Adjusted R2 window', layout, margins=(45,30))
        event, values=window.read()
        window.close()

def test_fxn(model, scale_flag, tune_key, test_df, dvfSVR):
    pred_test=tb.test_pred(model, scale_flag, fp, test_df, imp_flag)
    layout=[[sg.Text('\nPrediction for testing set completed !\n\nExit this window to continue\n', text_color='blue', font='Helvetica')]]
    window=sg.Window('Testing update window', layout, margins=(30,25))
    event, values=window.read()
    window.close()
    return (pred_test)

def export_fxn(model, scale_flag, tune_key):
    tb.export_model(model, scale_flag, tune_key)
    layout=[[sg.Text('\nModel exported successfully !\n\nExit this window to continue\n', text_color='blue', font='Helvetica')]]
    window=sg.Window('Model export window', layout, margins=(30,25))
    event, values=window.read()
    window.close()


def line_plot():
    df, data, header_list, fn, r1=which_set()
    num_cols, cat_cols =tb.num_cat_cols(df)
    col_for_line_y=num_cols.copy()
    #col_for_line.append('All columns')
    col_for_line_x=num_cols.copy()
    col_for_line_x.insert(0, 'Over index')
    layout=[
            [sg.Text('')],
            [sg.Text('Parameter on X-axis:\t'),
             sg.Combo(col_for_line_x, enable_events=True, key='combo_line_x')],
            [sg.Text('Parameter on Y-axis:\t'),
             sg.Combo(col_for_line_y, enable_events=True, key='combo_line_y')],
            [sg.Text('--'*45)],
            [sg.Checkbox('Sort by column:\t', key='cb_line_sb', default=False),
             sg.Combo(cat_cols, enable_events=True, key='combo_line_sort')],
            [sg.Text('--'*45)],
            [sg.Button('OK')],
            [sg.Text('')]
           ]
    window=sg.Window('LinePlot window', layout)
    while True:
        event, values = window.read()
        if event==sg.WIN_CLOSED:
            break
        if event=='OK':
            x=values['combo_line_x']
            y=values['combo_line_y']
            if values['cb_line_sb']==True:
                hue=values['combo_line_sort']
            else:
                hue=None
            flag=tb.line_plot(df, x, y, hue)
            if flag==1:
                sg.popup_error('\nERROR\n\nEnsure that the sort column has no NaN values\n', font='Helvetica', text_color='black')
                continue
    window.close()

def bar_plot():
    df, data, header_list, fn, r1=which_set()
    index_list, column_list=tb.index_column_list(df)
    num_cols, cat_cols =tb.num_cat_cols(df)
    '''layout=[
            [sg.Text('\nParameter on X-axis:', text_color='blue', font='Helvetica')],
            [sg.Radio('from index\t', 'bar_X', default=False, key='radio_ix'),
             sg.Combo(index_list, enable_events=True, key='combo_ix')],
            [sg.Radio('from column\t', 'bar_X', default=False, key='radio_cx'),
             sg.Combo(column_list, enable_events=True, key='combo_cx')],
            [sg.Text('--'*45)],
            [sg.Text('\nParameter on Y-axis:', text_color='blue', font='Helvetica')],
            [sg.Radio('from index\t', 'bar_Y', default=False, key='radio_iy'),
             sg.Combo(index_list, enable_events=True, key='combo_iy')],
            [sg.Radio('from column\t', 'bar_Y', default=False, key='radio_cy'),
             sg.Combo(column_list, enable_events=True, key='combo_cy')],
            [sg.Text('--'*45)],
            [sg.Text('')],
            [sg.Checkbox('Sort by column:', default=False, key='cb_bar_sort'),
             sg.Combo(cat_cols, enable_events=True, key='combo_bar_sort')],
            [sg.Text('--'*45)],
            [sg.Button('OK')],
            [sg.Text('')]
           ]'''
    layout=[
            [sg.Text('')],
            [sg.Text('Parameter on X-axis:\t'),
             sg.Combo(cat_cols, enable_events=True, key='combo_bar_x')],
            [sg.Text('Parameter on Y-axis:\t'),
             sg.Combo(num_cols, enable_events=True, key='combo_bar_y')],
            [sg.Text('--'*45)],
            [sg.Checkbox('Sort by column:\t', key='cb_bar_sb', default=False),
             sg.Combo(cat_cols, enable_events=True, key='combo_bar_sort')],
            [sg.Text('--'*45)],
            [sg.Button('OK')],
            [sg.Text('')]
           ]
    window=sg.Window('Bar-Plot window', layout)
    while True:
        #fx=0
        #fy=0
        event, values=window.read()
        if event==sg.WIN_CLOSED:
            break
        if event=='OK':
            x=values['combo_bar_x']
            y=values['combo_bar_y']
            if values['cb_bar_sb']==True:
                hue=values['combo_bar_sort']
            else:
                hue=None
            '''if values['radio_ix']==True:
                fx=1
                x_par=values['combo_ix']
            if values['radio_cx']==True:
                fx=2
                x_par=values['combo_cx']
            if values['radio_iy']==True:
                fy=1
                y_par=values['combo_iy']
            if values['radio_cy']==True:
                fy=2
                y_par=values['combo_cy']'''
            #bar_error=tb.bar_plot(df, x_par, y_par, hue, fx, fy)
            bar_error=tb.bar_plot(df, x, y, hue)
            if bar_error==1:
                #sg.popup_error('Error: Invalid parameters')
                sg.popup_error('\nERROR\n\nEnsure that the sort column has no NaN values\n', font='Helvetica', text_color='black')

                continue
    window.close()

def heatmap_plot():
    df, data, header_list, fn, r1=which_set()
    layout=[
            [sg.Text('\nCreate Heatmap for:', font='Helvetica', text_color='blue')],
            [sg.Radio('Complete dataset', 'heatmap', default=False, key='hm1')],
            [sg.Radio('Correlation matrix', 'heatmap', default=False, key='hm2')],
            [sg.Text('')],
            [sg.Button('OK')],
            [sg.Text('')]
           ]
    window=sg.Window('Heatmap creation window', layout, margins=(30,25))
    while True:
        fhm=0
        event, values=window.read()
        if event==sg.WIN_CLOSED:
            break
        if event=='OK':
            if values['hm1']==True:
                fhm=1
            else:
                fhm=2
            hm_error=tb.heatmap_plot(fhm, df)
            if hm_error==1:
                sg.popup_error('\nError: Categorical variables found\n')
                continue
    window.close()

def scatter_plot():
    df, data, header_list, fn, r1=which_set()
    num_cols, cat_cols =tb.num_cat_cols(df)
    layout=[
            [sg.Text('')],
            [sg.Text('Parameter on X-axis:\t'),
             sg.Combo(num_cols, enable_events=True, key='combo_scatter_x')],
            [sg.Text('Parameter on Y-axis:\t'),
             sg.Combo(num_cols, enable_events=True, key='combo_scatter_y')],
            [sg.Text('--'*45)],
            [sg.Checkbox('Sort by column:\t', key='cb_scatter_sb', default=False),
             sg.Combo(cat_cols, enable_events=True, key='combo_scatter_sort')],
            [sg.Checkbox('Keep regression line in plot', key='cb_scatter_kr', default=False)],
            [sg.Text('--'*45)],
            [sg.Button('OK')],
            [sg.Text('')]
           ]
    window=sg.Window('ScatterPlot window', layout)
    while True:
        event, values=window.read()
        if event==sg.WIN_CLOSED:
            break
        if event=='OK':
            skr=False
            x=values['combo_scatter_x']
            y=values['combo_scatter_y']
            # Sort or not ?
            if values['cb_scatter_sb']==True:
                hue=values['combo_scatter_sort']
            else:
                hue=None
            # Keep Reg line or not ?
            if values['cb_scatter_kr']==True:
                skr=True
            flag=tb.scatter_plot(df, x, y, hue, skr)
            if flag==1:
                sg.popup_error('\nERROR\n\nEnsure that the sort column has no NaN values\n', font='Helvetica', text_color='black')
    window.close()

def swarm_plot():
    df, data, header_list, fn, r1=which_set()
    num_cols, cat_cols =tb.num_cat_cols(df)
    layout=[
            [sg.Text('')],
            [sg.Text('Parameter on X-axis:'),
             sg.Combo(cat_cols, enable_events=True, key='combo_swarm_x')],
            [sg.Text('Parameter on Y-axis:'),
             sg.Combo(num_cols, enable_events=True, key='combo_swarm_y')],
            [sg.Text('--'*45)],
            [sg.Checkbox('Sort by column:', key='cb_swarm_sb', default=False),
             sg.Combo(cat_cols, enable_events=True, key='combo_swarm_sort')],
            [sg.Text('--'*45)],
            [sg.Button('OK')],
            [sg.Text('')]
           ]
    window=sg.Window('Swarm Plot window', layout)
    while True:
        event, values=window.read()
        if event==sg.WIN_CLOSED:
            break
        if event=='OK':
            x=values['combo_swarm_x']
            y=values['combo_swarm_y']
            hue=None
            if values['cb_swarm_sb']==True:
                hue=values['combo_swarm_sort']
            flag=tb.swarm_plot(df, x, y, hue)
            if flag==1:
                sg.popup_error('\nERROR\n\nEnsure that the sort column has no NaN values\n', font='Helvetica', text_color='black')
                continue
    window.close()

def box_plot():
    df, data, header_list, fn, r1=which_set()
    num_cols, cat_cols =tb.num_cat_cols(df)
    layout=[
            [sg.Text('')],
            [sg.Text('Parameter on X-axis:'),
             sg.Combo(cat_cols, enable_events=True, key='combo_box_x')],
            [sg.Text('Parameter on Y-axis:'),
             sg.Combo(num_cols, enable_events=True, key='combo_box_y')],
            [sg.Text('--'*45)],
            [sg.Checkbox('Sort by column:', key='cb_box_sb', default=False),
             sg.Combo(cat_cols, enable_events=True, key='combo_box_sort')],
            [sg.Text('--'*45)],
            [sg.Button('OK')],
            [sg.Text('')]
           ]
    window=sg.Window('Box Plot window', layout)
    while True:
        event, values=window.read()
        if event==sg.WIN_CLOSED:
            break
        if event=='OK':
            x=values['combo_box_x']
            y=values['combo_box_y']
            hue=None
            if values['cb_box_sb']==True:
                hue=values['combo_box_sort']
            flag=tb.box_plot(df, x, y, hue)
            if flag==1:
                sg.popup_error('\nERROR\n\nEnsure that the sort column has no NaN values\n', font='Helvetica', text_color='black')
                continue
    window.close()

def hist_plot():
    df, data, header_list, fn, r1=which_set()
    num_cols, cat_cols =tb.num_cat_cols(df)
    layout=[
            [sg.Text('')],
            [sg.Text('Parameter on X-axis:'),
             sg.Combo(header_list, enable_events=True, key='combo_hist_x')],
            [sg.Text('')],
            [sg.Checkbox('Sort by column:', default=False, key='cb_hist_sort'),
             sg.Combo(cat_cols, enable_events=True, key='combo_hist_sort')],
            [sg.Text('')],
            [sg.Button('OK')],
            [sg.Text('')]
           ]
    window=sg.Window('Histogram window', layout)
    while True:
        event, values=window.read()
        if event==sg.WIN_CLOSED:
            break
        if event=='OK':
            x=values['combo_hist_x']
            hue=None
            if values['cb_hist_sort']==True:
                hue=values['combo_hist_sort']
            flag=tb.hist_plot(df, x, hue)
            if flag==1:
                sg.popup_error('\nERROR\n\nEnsure that the sort column has no NaN values\n', font='Helvetica', text_color='black')
                continue
    window.close()

def density_plot(): #DISCONTINUED
    df, data, header_list, fn, r1=which_set()
    num_cols, cat_cols =tb.num_cat_cols(df)
    layout=[
            [sg.Text('')],
            [sg.Text('Parameter on X-axis:'),
             sg.Combo(num_cols, enable_events=True, key='combo_kde_x')],
            [sg.Text('')],
            [sg.Checkbox('Sort by column:', default=False, key='cb_kde_sort'),
             sg.Combo(cat_cols, enable_events=True, key='combo_kde_sort')],
            [sg.Text('')],
            [sg.Button('OK')],
            [sg.Text('')]
           ]
    window=sg.Window('Density plot window', layout)
    while True:
        event, values=window.read()
        if event==sg.WIN_CLOSED:
            break
        if event=='OK':
            x=values['combo_kde_x']
            hue=None
            if values['cb_kde_sort']==True:
                hue=values['combo_kde_sort']
            tb.density_plot(df, x, hue)
    window.close()

def join_plot(): #Contour
    df, data, header_list, fn, r1=which_set()
    num_cols, cat_cols =tb.num_cat_cols(df)
    layout=[
            [sg.Text('')],
            [sg.Text('Parameter on X-axis:\t'),
             sg.Combo(num_cols, enable_events=True, key='combo_join_x')],
            [sg.Text('Parameter on Y-axis:\t'),
             sg.Combo(num_cols, enable_events=True, key='combo_join_y')],
            [sg.Text('--'*45)],
            [sg.Checkbox('Sort by column:', key='cb_join_sb', default=False),
             sg.Combo(cat_cols, enable_events=True, key='combo_join_sort')],
            [sg.Text('--'*45)],
            [sg.Button('OK')],
            [sg.Text('')]
           ]
    window=sg.Window('2D KDE Plot window', layout)
    while True:
        event, values=window.read()
        if event==sg.WIN_CLOSED:
            break
        if event=='OK':
            x=values['combo_join_x']
            y=values['combo_join_y']
            hue=None
            if values['cb_join_sb']==True:
                hue=values['combo_join_sort']
            flag=tb.join_plot(df, x, y, hue)
            if flag==1:
                sg.popup_error('\nERROR\n\nEnsure that the sort column has no NaN values\n', font='Helvetica', text_color='black')
                continue
    window.close()

def count_plot(): #DISCONTINUED
    df, data, header_list, fn, r1=which_set()
    num_cols, cat_cols =tb.num_cat_cols(df)
    layout=[
            [sg.Text('')],
            [sg.Text('Parameter on X-axis:'),
             sg.Combo(cat_cols, enable_events=True, key='combo_count_x')],
            [sg.Text('')],
            [sg.Checkbox('Sort by column:', default=False, key='cb_count_sort'),
             sg.Combo(cat_cols, enable_events=True, key='combo_count_sort')],
            [sg.Text('')],
            [sg.Button('OK')],
            [sg.Text('')]
           ]
    window=sg.Window('Count plot window', layout)
    while True:
        event, values=window.read()
        if event==sg.WIN_CLOSED:
            break
        if event=='OK':
            x=values['combo_count_x']
            hue=None
            if values['cb_count_sort']==True:
                hue=values['combo_count_sort']
            tb.count_plot(df, x, hue)
    window.close()
# Definitions End here...

# Final layout START...
logo_and_name = [
        [sg.Image(key='logo1', filename=logo_path),
         sg.Text('\t'),
         sg.Text('Machine Learning Toolkit', font=('Calibri', 25, ' bold'), text_color='grey')],
        [sg.Text('_'*112)]
       ]

input_col = [
             [sg.Text('\tTraining Dataset:\t'),
              sg.In(enable_events=True, key='inp1'),
              sg.FileBrowse(file_types=(('CSV Files', '*.csv'),)),
              sg.Checkbox('Has header names?', key='cb1', default=True)],
             [sg.Text('\tTesting Dataset:\t'),
              sg.In(enable_events=True, key='inp2'),
              sg.FileBrowse(file_types=(('CSV Files', '*.csv'),)),
              sg.Checkbox('Has header names?', key='cb2', default=True)]
            ]

set_selection_column = [
                        [sg.Text('\t\tSelect Dataset for Analysis:\t'),
                         sg.Radio('Training Dataset\t', 'RADIO1', enable_events=True, default=False, key='r1'),
                         sg.Radio('Testing Dataset', 'RADIO1', default=False, key='r2')]
                       ]

buts_col = [
            [sg.Text('\t'),
            sg.Button('Data', tooltip=' Shows the entire dataset '),
            sg.Text(),
            sg.Button('Head', tooltip=' Shows first 5 rows of the dataset '),
            sg.Text(),
            sg.Button('Tail', tooltip=' Shows last 5 rows of the dataset '),
            sg.Text(),
            sg.Button('Shape', tooltip=' Shows (total rows, total columns) '),
            sg.Text(),
            sg.Button('Describe', tooltip=' Shows statistical data '),
            sg.Text(),
            sg.Button('Info', tooltip=' Shows column dtypes & NaN count ')]
           ]

cleaning_col = [
                [sg.Text('\t'),
                sg.Button('Drop', tooltip=' Drop rows or columns '),
                sg.Text(),
                sg.Button('Replace', tooltip=' Replace values in columns '),
                sg.Text(),
                sg.Button('Fill NaN', tooltip=' Fill missing NaN values '),
                sg.Text(),
                sg.Button(' Convert dtype ', tooltip=' Convert dtype of a column '),
                sg.Text(),
                #sg.Button('Unique', tooltip=' Finds unique entries in a column '),
                #sg.Text(),
                sg.Button('Value count', tooltip=' Checks the count of unique entries in a column '),
                sg.Text(),
                sg.Button(' Label encode ', tooltip=' Encode the categorical column labels into numerical values ')]
               ]

plot_col = [
            [sg.Text('\t'),
             sg.Button('Line'),
             sg.Text(),
             sg.Button('Bar'),
             sg.Text(),
             sg.Button('Box'),
             sg.Text(),
             sg.Button('Contour')],
            [sg.Text('\t'),
             sg.Button('Scatter'),
             sg.Text(),
             sg.Button('Heatmap'),
             sg.Text(),
             sg.Button('Swarm'),
             sg.Text(),
             sg.Button('Histogram')]
           ]

build_model_col = [
                 [sg.Text('\t'),
                  sg.Button('Model', tooltip=' Select the target variable, split validation & training set & apply the model algorithm '),
                  sg.Text(),
                  sg.Button('Tune', tooltip=' Tune the parameters for the selected algorithm '),
                  sg.Text(),
                  sg.Button('Validate', tooltip=' Fit the model on training set and predict for VALIDATION set '),
                  sg.Text(),
                  sg.Button('Metric', tooltip=' Select an appropriate metric option to analyse the predictions '),
                  sg.Text(),
                  sg.Button('Export', tooltip=' Save and export the model for further use ')]
                ]

consume_model_col = [
                     [sg.Text('\t'),
                      sg.Button('Import', tooltip=' Import an already saved model(in pickle format) '),
                      sg.Text(),
                      sg.Button('Test', tooltip=' Fit the model on training set and predict for TESTING set ')],
                    ]

layout = [
          [sg.Text(' Import Data', font=('Calibri', 14, 'bold'), text_color='blue')],
          [sg.Column(input_col)],
          [sg.Column(set_selection_column)],
          [sg.Text('_'*112)],
          [sg.Text(' View Data', font=('Calibri', 14, 'bold'), text_color='blue')],
          [sg.Column(buts_col)],
          [sg.Text('_'*112)],
          [sg.Text(' Clean Data', font=('Calibri', 14, 'bold'), text_color='blue')],
          [sg.Column(cleaning_col)],
          [sg.Text('_'*112)],
          [sg.Text(' Analyse Data', font=('Calibri', 14, 'bold'), text_color='blue')],
          [sg.Column(plot_col)],
          [sg.Text('_'*112)],
          [sg.Text(' Train model', font=('Calibri', 14, 'bold'), text_color='blue')],
          [sg.Column(build_model_col)],
          [sg.Text('_'*112)],
          [sg.Text(' Test model', font=('Calibri', 14, 'bold'), text_color='blue')],
          [sg.Column(consume_model_col)],
          [sg.Text('_'*112)],
         ]

copy_right = [
              [sg.Text('       Copy Right Intelimek Systems\n', font=('Ariel', 13, ' bold'), text_color='grey')]
             ]

final_layout = [
                [sg.Column(logo_and_name)],
                [sg.Column(layout, size=(800, 480), scrollable=True)],
                [sg.Column(copy_right, justification='center')]
               ]

# Final layout END...

window = sg.Window('Desktop_Toolkit', layout=final_layout)
#window.maximize()

header_check_1=False
header_check_2=False
tc=0
while True:
    event, values=window.read()
    r1=0
    if event==sg.WIN_CLOSED:
        break
    if event=='inp1':
        try: # train_df, train_data, train_header_list=read_data(fp1, header_check_1)
            if values['cb1']==True:
                header_check_1=True
            fp1=values['inp1']
            train_df, train_data, train_header_list=read_data(fp1, header_check_1)
            fn1=fp1.split('/')[-1]
            #window['t1'].update("1) Training Dataset: {}".format(fn1))
            train_folder_path=tb.training_data_folder_creation(train_df)
            inp_dict=tb.tracker_id_dict()
            inp_dict=tb.tracker_dict_mod(inp_dict, 'training_set', fn1)
            tb.tracker_add_list(inp_list, 'input', inp_dict)
        except:
            continue
    if event=='inp2':
        try: # test_df, test_data, test_header_list=read_data(fp2, header_check_2)
            if values['cb2']==True:
                header_check_2=True
            fp2=values['inp2']
            test_df, test_data, test_header_list=read_data(fp2, header_check_2)
            fn2=fp2.split('/')[-1]
            #window['t2'].update("2) Testing Dataset: {}".format(fn2))
            test_folder_path=tb.testing_data_folder_creation(test_df)
            inp_dict=tb.tracker_id_dict()
            inp_dict=tb.tracker_dict_mod(inp_dict, 'testing_set', fn2)
            tb.tracker_add_list(inp_list, 'input', inp_dict)
        except:
            continue

    if event=='Data':
        try: # data_show()
            data_show()
        except:
            continue
    if event=='Head':
        try: # head_show()
            head_show()
        except:
            continue
    if event=='Tail':
        try: # tail_show()
            tail_show()
        except:
            continue
    if event=='Describe':
        try: # describe_show()
            describe_show()
        except:
            continue
    if event=='Shape':
        try: # shape_show()
            shape_show()
        except:
            continue
    if event=='Info':
        try: # info_show()
            info_show()
        except:
            continue

    if event=='Drop':
        try: # updated_df, updated_header, r1=drop_window()
            updated_df, updated_header, r1=drop_window()
            if r1==1:   # train_df
                train_df=updated_df
                train_header_list=updated_header
                tb.after_drop_folder_updation_train(train_folder_path, train_df)
                tb.tracker_add_list(EDA_1_list, 'eda_train', 'drop')
            elif r1==0:     # test_df
                test_df=updated_df
                test_header_list=updated_header
                tb.after_drop_folder_updation_test(test_folder_path, test_df)
                tb.tracker_add_list(EDA_2_list, 'eda_test', 'drop')
        except:
            continue
    if event=='Replace':
        try: # r1=replace_window()
            r1=replace_window()
            if r1==1:   # train_df
                tb.after_replace_folder_updation_train(train_folder_path, train_df)
                tb.tracker_add_list(EDA_1_list, 'eda_train', 'replace')
            elif r1==0:       # test_df
                tb.after_replace_folder_updation_test(test_folder_path, test_df)
                tb.tracker_add_list(EDA_2_list, 'eda_test', 'replace')
        except:
            continue
    if event=='Fill NaN':
        try: # r1=fillna_window()
            r1=fillna_window()
            if r1==1:   # train_df
                tb.after_fillna_folder_updation_train(train_folder_path, train_df)
                tb.tracker_add_list(EDA_1_list, 'eda_train', 'fill')
            elif r1==0:       # test_df
                tb.after_fillna_folder_updation_test(test_folder_path, test_df)
                tb.tracker_add_list(EDA_2_list, 'eda_test', 'fill')
        except:
            continue
    if event==' Convert dtype ':
        try: # r1=convert_window()
            r1=convert_window()
            if r1==1:   # train_df
                tb.after_convert_folder_updation_train(train_folder_path, train_df)
                tb.tracker_add_list(EDA_1_list, 'eda_train', 'convert')
            elif r1==0:       # test_df
                tb.after_convert_folder_updation_test(test_folder_path, test_df)
                tb.tracker_add_list(EDA_2_list, 'eda_test', 'convert')
        except:
            continue
    if event=='Unique': # DISCONTINUED
        try: # unique_window()
            unique_sel_window()
        except:
            continue
    if event=='Value count':
        try: # value_counts_window()
            value_counts_window()
        except:
            continue
    if event==' Label encode ':
        try: # r1=label_encode_window()
            r1=label_encode_window()
            if r1==1:   # train_df
                tb.after_encode_folder_updation_train(train_folder_path, train_df)
                tb.tracker_add_list(EDA_1_list, 'eda_train', 'encode')
            elif r1==0:       # test_df
                tb.after_encode_folder_updation_test(test_folder_path, test_df)
                tb.tracker_add_list(EDA_2_list, 'eda_test', 'encode')
        except:
            continue

    if event=='Import':
        try: # fp, model=import_window()
            fp, model=import_window()
            imp_flag=1
            mn=fp.split('/')[-1]
            scale_flag=mn.split('_')[0]
            tune_key=mn.split('_')[1]
            algo=None
            tb.tracker_add_status('import')
            print(scale_flag, tune_key)
        except:
            continue
    if event=='Model':
        try: # model, algo, tvs_flag, scale_flag, tune_key, X_train, X_val, y_train, y_val=model_window(train_df)
            tc=tc+1
            model, algo, tvs_flag, scale_flag, tune_key, X_train, X_val, y_train, y_val=model_window(train_df)
            imp_flag=0
            fp=None

            print('TC:', tc)
        except:
            continue
    if event=='Tune':
        try: # model=tune_window(model, algo, tune_key, X_train, y_train)
            model=tune_window(model, algo, tune_key, X_train, y_train)
            tb.tracker_add_dict('tune', tune_list, tc)
            print(model)
        except:
            continue
    if event=='Validate':
        try: # pred_val, pred_train, pvf=validate_fxn(model, tvs_flag, tune_key, X_val, X_train, dvfSVR)
            dvfSVR=None # Too lazy to remove
            pred_val, pred_train, pvf=validate_fxn(model, tvs_flag, tune_key, X_val, X_train, dvfSVR)

            if pvf==1:
                tb.training_pred_folder_creation(pred_train, X_train, y_train, train_folder_path)
                tb.validation_pred_folder_creation(pred_val, X_val, y_val, train_folder_path)
            else:
                tb.training_pred_folder_creation(pred_train, X_train, y_train, train_folder_path)
            tb.tracker_add_dict('validate', val_list, tc)
            #tb.tracker_add_list(val_list, 'validate', 'run_id:{}...done'.format(tc))
        except:
            continue
    if event=='Metric':
        try: # result_train, result_val, mf=tb.metric_scoring(tvs_flag, metric, model, pred_val, X_val, y_val, pred_train, X_train, y_train)
            metric=metric_selection_window()
            result_train, result_val, mf=tb.metric_scoring(tvs_flag, metric, model, pred_val, X_val, y_val, pred_train, X_train, y_train)
            if mf==None:
                pass
            else:
                metric_display(result_train, result_val, mf)
        except:
            continue
    if event=='Test':
        try: # pred_test = test_fxn(model, scale_flag, tune_key, test_df, dvfSVR)
            if values['inp2']!='':
                dvfSVR=None
                pred_test = test_fxn(model, scale_flag, tune_key, test_df, dvfSVR)
            else:
                sg.popup_error('\nError\n\nBrowse the testing dataset first\n', font='Helvetica', text_color='blue')
                continue
            tb.testing_pred_folder_creation(pred_test, test_df, test_folder_path)
            tb.tracker_add_status('test')
        except:
            continue
    if event=='Export':
        try: # export_fxn(model, scale_flag, tune_key)
            export_fxn(model, scale_flag, tune_key)
            tb.tracker_add_dict('export', exp_list, tc)
            #tb.tracker_add_status('export')
        except:
            continue


    if event=='Line':
        try:
            line_plot()
        except:
            continue
    if event=='Bar':
        try:
            bar_plot()
        except:
            continue
    if event=='Heatmap':
        try:
            heatmap_plot()
        except:
            continue
    if event=='Scatter':
        try:
            scatter_plot()
        except:
            continue
    if event=='Swarm':
        try:
            swarm_plot()
        except:
            continue
    if event=='Box':
        try:
            box_plot()
        except:
            continue
    if event=='Histogram':
        try:
            hist_plot()
        except:
            continue
    if event=='Contour':
        try:
            join_plot()
        except:
            continue

    ''' # DISCONTINUED PART
    if event=='Density':
        try:
            density_plot()
        except:
            continue
    if event=='Count':
        try:
            count_plot()
        except:
            continue
    '''
window.close()
