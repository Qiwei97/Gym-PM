import pandas as pd
import numpy as np

def load_data(Type='PdM1'):

    """
        Custom dataset must include:
            age: time since last failure
            Failure: 1 = failed, 0 = did not fail
            ttf: time to next failure (For display purposes, would not be fed into the model)
            Date: must be sorted sequentially
    """

    file_path = 'Gym-PM/gym_pm/data/' + Type + '.csv'
    df = pd.read_csv(file_path)

    if Type == 'PdM1':

        df = df.iloc[:, :-7]
        df['age'] = df['Hours Since Previous Failure']
        df.Date = pd.to_datetime(df.Date)
        df = df.sort_values('Date')
        df = df.drop(columns = ['ID', 'Operator', 'Hours Since Previous Failure', 'Date'])
        df.Failure = df.Failure.apply(lambda x: 0 if x == 'No' else 1)
        df = df[~((df.age <= 5) & (df.Failure == 1))]
        df.reset_index(drop = True, inplace = True)

        # ttf
        failure_time = np.array(sorted(df[df.Failure == 1].index.tolist()))
        failure_list = []
        for i in range(len(df)):
            failure_list.append(failure_time)

        df['ttf'] = failure_list
        df.ttf = df.ttf - df.index
        df.ttf = df.ttf.apply(lambda x: x[x >= 0]) # Drop negative values
        df = df[df.ttf.str.len() > 0] # Drop empty lists
        df.ttf = df.ttf.apply(lambda x: x[0])
        df.reset_index(drop = True, inplace = True)

    elif Type == 'PdM2':

        df['Failure'] = df.Failure_today.apply(lambda x: 0 if x == 'No' else 1)
        df.Date = pd.to_datetime(df.Date)
        df = df[df.Date > '2017']
        df = df.sort_values('Date')
        df.drop(columns = ['Fail_tomorrow', 'Failure_today', 'Location', 'Date', 
                           'Parameter1_Dir', 'Parameter2_9am', 'Parameter2_3pm'], inplace = True)
        df.fillna(0, inplace = True)
        df.reset_index(drop = True, inplace = True)

        # ttf
        failure_time = np.array(sorted(df[df.Failure == 1].index.tolist()))
        failure_list = []
        for i in range(len(df)):
            failure_list.append(failure_time)

        df['ttf'] = failure_list
        df.ttf = df.ttf - df.index
        df['age'] = df.ttf
        df.ttf = df.ttf.apply(lambda x: x[x >= 0]) # Drop negative values
        df = df[df.ttf.str.len() > 0] # Drop empty lists
        df.ttf = df.ttf.apply(lambda x: x[0])

        # age
        df.age = df.age.apply(lambda x: x[x < 0]) # Drop positive values
        df = df[df.age.str.len() > 0] # Drop empty lists
        df.age = df.age.apply(lambda x: -x[-1])
        df = df[~((df.age <= 5) & (df.Failure == 1))]
        df.reset_index(drop = True, inplace = True)

    return df


