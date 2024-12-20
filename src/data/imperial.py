import os
import numpy as np
import pandas as pd
from tabula import read_pdf


def read_both_2022(f='raw_data/Exam_results_PDF_2022.pdf', pages = ['33', '34', '35',
                                                                    '36', '37', '38']):
    
    def read_pdf_2022(f, page):
        df = read_pdf(f, pages=page)[0]
        new_df = df.drop(columns=['Name', 'Unnamed: 0', 'Decile', 'Part I', 'Result'])
        new_df = new_df.tail(-2)
        return new_df
    
    df = read_pdf_2022(f, pages[0])
    for page in pages[1:]:
        next_df = read_pdf_2022(f, page)
        if page == pages[-1]:
            # remove averaging from last row of last page
            next_df = next_df.head(-1)
        
        df = pd.concat([df, next_df], ignore_index=True)
        
    a = df['ELEC40010+11(maths)'].str.split(pat=' ', n=3, expand=True)
    a.columns = ['40010-11-MTs', '40010-11-MdA', '40010-11-MdB', '40010-11-Tot']
    df = pd.merge(left=df, right=a, left_index=True, right_index=True)

    d = df['ELEC40009  ELEC40006'].str.split(pat=' ', n=1, expand=True)
    d.columns = ['ELEC40009', 'ELEC40006']
    df = pd.merge(left=df, right=d, left_index=True, right_index=True)

    df = df.drop(columns=['ELEC40010+11(maths)',
                                  '40010-11-MTs',
                                  '40010-11-Tot',
                                  'ELEC40002',
                                  'ELEC40003',
                                  'ELEC40004',
                                  'ELEC40009  ELEC40006'])

    df= df.rename(columns={'Unnamed: 1': 'ELEC40002',
                            'Unnamed: 2': 'ELEC40003',
                            'Unnamed: 3': 'ELEC40004'})
    
    # add students who are resitting second year - use their first year results from 2021
    new_row = {'No.': '01890431', 'ELEC40002': 50, 'ELEC40003': 52, 'ELEC40004': 63,
            '40010-11-MdA': 44, '40010-11-MdB': 40, 'ELEC40009': 47, 
            'ELEC40006': 57.08}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    new_row = {'No.': '01929788', 'ELEC40002': 53, 'ELEC40003': 59, 'ELEC40004': 40,
            '40010-11-MdA': 64, '40010-11-MdB': 42, 'ELEC40009': 40, 
            'ELEC40006': 63.47}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # clean
    cid_clean = df['No.'].astype(str).apply(lambda x: x[1:] if x.startswith('0') else x)
    cid_clean = cid_clean.apply(lambda x: x[:-4] if x.endswith(' (R)') else x)
    cid_clean = cid_clean.apply(lambda x: x[:-2] if x.endswith('.0') else x)
    df['No.'] = cid_clean.astype(int)

    # replace absents with zeros
    df = df.replace('Abs', 0)

    # Remove NaNs
    df = df[df.isnull().sum(axis=1) < 2]
    df = df.fillna(0)
    df.shape

    xl_file = pd.ExcelFile('raw_data/second year tutorials for algorithm.xlsx')
    dfs = {sheet_name: xl_file.parse(sheet_name) 
            for sheet_name in xl_file.sheet_names}

    df = dfs['EEE'].merge(df, left_on='CID', right_on='No.')

    # add students who changed stream last minute 
    idx = np.where(df['CID'] == 2020841)[0][0]
    df.loc[idx, 'Stream '] = 'EEE'

    # group by student preference: EIE or EEE
    df_eee = df[df['Stream '] == 'EEE'][['No.',
        'ELEC40002', 'ELEC40003', 'ELEC40004', '40010-11-MdA', '40010-11-MdB',
        'ELEC40009', 'ELEC40006']]

    df_eie = df[df['Stream '] == 'EIE'][['No.',
        'ELEC40002', 'ELEC40003', 'ELEC40004', '40010-11-MdA', '40010-11-MdB',
        'ELEC40009', 'ELEC40006']]
    return df_eee, df_eie


def load_data(path, load_data_fn_name, load_data_fn_kwargs):
    os.makedirs(path, exist_ok=True)
    eee_path = path + 'eee.csv'
    eie_path = path + 'eie.csv'

    try:
        df_eee = pd.read_csv(eee_path)
        df_eie = pd.read_csv(eie_path)
        
    except FileNotFoundError:
        print('Data not found. Loading from raw data.')
        load_data_fn = globals()[load_data_fn_name]
        df_eee, df_eie = load_data_fn(**load_data_fn_kwargs)
        df_eee.to_csv(eee_path, index=False)
        df_eie.to_csv(eie_path, index=False)
        print('Done.')
    
    return df_eee, df_eie