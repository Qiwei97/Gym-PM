import pandas as pd

def load_data(Type='PdM1'):

    file_path = 'data/' + Type + '.csv'
    df = pd.read_csv(file_path)

    if Type == 'PdM1':
        df = df.iloc[:, :-7]
        df['age'] = df['Hours Since Previous Failure']
        df.Date = pd.to_datetime(df.Date)
        df = df.sort_values('Date')
        df = df.drop(columns=['ID', 'Operator', 'Hours Since Previous Failure', 'Date'])
        df.Failure = df.Failure.apply(lambda x: 0 if x == 'No' else 1)
        df = df[~((df.age == 1) & (df.Failure == 1))]
        return df

    if Type == 'PdM2':
        df['Failure'] = df.Failure_today.apply(lambda x: 0 if x == 'No' else 1)
        df.Date = pd.to_datetime(df.Date)
        df = df.sort_values('Date')
        df.drop(columns=['Fail_tomorrow', 'Failure_today', 'Location', 'Date'], inplace=True)
        df.fillna(0, inplace=True)
        return df
