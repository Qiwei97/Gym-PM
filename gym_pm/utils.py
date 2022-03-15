import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def create_data(Type='PdM2', split='Train', save=True):

    """
        Custom dataset must include:
            age: time since last failure
            Failure: 1 = failed, 0 = did not fail
            ttf: time to next failure (For display purposes, would not be fed into the model)
            Date: must be sorted sequentially
        
        Splits: Train, Test, None
    """

    file_path = 'Gym-PM/gym_pm/data/' + Type + '.csv'
    df = pd.read_csv(file_path)

    if Type == 'PdM2':

        df['Failure'] = df.Failure_today.apply(lambda x: 0 if x == 'No' else 1)
        df.Date = pd.to_datetime(df.Date)

        cutoff = '2016-10'
        if split == 'Train':
            df = df[(df.Date > '2015-03') & (df.Date < cutoff)]
        elif split == 'Test':
            df = df[df.Date >= cutoff]
        elif split == 'All':
            df = df[df.Date > '2015-03']
        else:
            return "Invalid Split"

        df = df.sort_values('Date')
        df.drop(columns = ['Fail_tomorrow', 'Failure_today', 'Parameter1_Dir', 
                           'Parameter2_9am', 'Parameter2_3pm'], inplace = True)
        df.fillna(0, inplace = True)
        df.reset_index(drop = True, inplace = True)

        # ttf
        failure_time = df.groupby(['Location', 'Date']).Failure.first() 
        failure_time = failure_time[failure_time == 1].reset_index() # Collect failure dates
        failure_time = failure_time.groupby('Location').Date.apply(np.array)
        failure_time = failure_time.reset_index()

        failure_time.rename(columns = {"Date": "ttf"}, inplace = True)
        df = df.merge(failure_time, how = 'inner', on = 'Location')
        df['age'] = df.ttf

        df.ttf = df.apply(lambda x: x['ttf'][x['ttf'] >= x['Date']], axis = 1)
        df = df[df.ttf.str.len() > 0] # Drop empty lists
        df.ttf = df.apply(lambda x: (x['ttf'][0] - x['Date']).days, axis = 1) # Calculate TTF

        # age
        df.age = df.apply(lambda x: x['age'][x['age'] < x['Date']], axis = 1)
        df = df[df.age.str.len() > 0] # Drop empty lists
        df.age = df.apply(lambda x: (x['Date'] - x['age'][-1]).days, axis = 1) # Calculate Age
        df = df[~(df.age + df.ttf <= 10)]
        df.reset_index(drop = True, inplace = True)
        df.drop(columns = ['Date', 'Location'], inplace = True)

        # Save data as pickle
        if save:
            df.to_pickle(file_path.replace('.csv', '_' + split + '.pkl'))

    return df


def update_boundaries(Type='PdM2', save=True,
                      file_path='Gym-PM/gym_pm/data/'):

    obs_bound = pd.DataFrame()
    df = create_data(Type, split='All', save=False)
    obs_bound['high'] = df.max()
    obs_bound['low'] = df.min()

    if save:
        obs_bound.to_pickle(file_path + Type + '_Bound.pkl')


def load_data(file_path, data, split):

    return pd.read_pickle(file_path + data + "_" + split + ".pkl")


def evaluate_baseline(eval_env, repair_policy=0, repair_interval=10, 
                      duration=None, resupply_threshold=None, 
                      output_threshold=2000, display=False):
    
    """
        repair_policy
        
        0: Repair when failed
        1: Repair at repair_interval
        
        Assembly_Env: Resupply triggered according to the resupply_threshold and backlog_threshold
    """
    
    if duration == None:
        duration = eval_env.max_duration
        
    assert duration <= eval_env.max_duration
    assert repair_interval < duration
    
    obs = eval_env.reset()
    action_list, obs_list = [], []
    
    # Rail Env
    if 'Rail' in str(eval_env):
    
        if repair_policy == 0:

            for i in range(duration):

                if obs['Failure'][0] == 1:
                    action = 0 # Repair
                else:
                    action = 1 # Do Nothing

                obs, reward, done, info = eval_env.step(action)
                    
                action_list.append(action)
                if display:
                    obs_list.append(eval_env.render('human').to_dict()['Results'])
                else:
                    obs_list.append(eval_env.render('console').to_dict()['Results'])

        elif repair_policy == 1:

            for i in range(duration):

                if (obs['Failure'][0] == 1) or (obs['age'][0] >= repair_interval):
                    action = 0 # Repair
                else:
                    action = 1 # Do Nothing

                obs, reward, done, info = eval_env.step(action)
                
                action_list.append(action)
                if display:
                    obs_list.append(eval_env.render('human').to_dict()['Results'])
                else:
                    obs_list.append(eval_env.render('console').to_dict()['Results'])

        else:

            print("Invalid Policy")
            return
        
    # Assembly Env
    elif 'Assembly' in str(eval_env):
        
        if resupply_threshold == None:
            resupply_threshold = eval_env.resupply_qty
    
        if repair_policy == 0:

            for i in range(duration):

                if obs['Failure'][0] == 1:
                    action = 0 # Repair
                elif ((obs['resources'][0] <= resupply_threshold) and (obs['output'][0] <= output_threshold)):
                    if len(eval_env.machine.resupply_list) == 0:
                        action = 1 # Resupply
                    else:
                        action = 2 # Do Nothing
                else:
                    action = 2 # Do Nothing                    
                    
                obs, reward, done, info = eval_env.step(action)
                
                action_list.append(action)
                if display:
                    obs_list.append(eval_env.render('human').to_dict()['Results'])
                else:
                    obs_list.append(eval_env.render('console').to_dict()['Results'])

        elif repair_policy == 1:

            for i in range(duration):

                if (obs['Failure'][0] == 1) or (obs['age'][0] >= repair_interval):
                    action = 0 # Repair
                elif ((obs['resources'][0] <= resupply_threshold) and (obs['output'][0] <= output_threshold)):
                    if len(eval_env.machine.resupply_list) == 0:
                        action = 1 # Resupply
                    else:
                        action = 2 # Do Nothing
                else:
                    action = 2 # Do Nothing
                    
                obs, reward, done, info = eval_env.step(action)
                
                action_list.append(action)
                if display:
                    obs_list.append(eval_env.render('human').to_dict()['Results'])
                else:
                    obs_list.append(eval_env.render('console').to_dict()['Results'])

        else:

            print("Invalid Policy")
            return
        
    else:
        
        print("Invalid Env Type. Must be either Rail or Assembly.")
        return
    
    df_result = pd.DataFrame(obs_list)
    df_result['action'] = action_list
    
    return df_result


def evaluate_policy(eval_env, trainer,
                    duration=None, RNN=False,
                    display=False):
    
    if duration == None:
        duration = eval_env.max_duration
        
    assert duration <= eval_env.max_duration
    
    obs = eval_env.reset()
    action_list, obs_list = [], []
    
    if RNN == False:

        for i in range(duration):                

            action = trainer.compute_action(obs)
            obs, reward, done, info = eval_env.step(action)
            
            action_list.append(action)
            if display:
                obs_list.append(eval_env.render('human').to_dict()['Results'])
            else:
                obs_list.append(eval_env.render('console').to_dict()['Results'])           
                
    else:
        
        # If use LSTM
        state = trainer.get_policy().model.get_initial_state()
        
        if 'Rail' in str(eval_env):
            action, reward = 1.0, 0.0
        elif 'Assembly' in str(eval_env):
            action, reward = 2.0, 0.0

        for i in range(duration):

            action, state, logit = trainer.compute_action(obs, prev_action=action,
                                                          prev_reward=reward, state=state)
            obs, reward, done, info = eval_env.step(action)

            action_list.append(action)
            if display:
                obs_list.append(eval_env.render('human').to_dict()['Results'])
            else:
                obs_list.append(eval_env.render('console').to_dict()['Results'])
    
    df_result = pd.DataFrame(obs_list)
    df_result['action'] = action_list
    
    return df_result


# Adapted from https://www.datahubbs.com/how-to-use-deep-reinforcement-learning-to-improve-your-supply-chain/
def plot_metrics(results):

    # Unpack values from each iteration
    rewards = np.hstack([i['hist_stats']['episode_reward'] for i in results])

    pol_loss = []
    vf_loss = []
    for i in results:
        metric = i['info']['learner']['default_policy']['learner_stats']
        pol_loss.append(metric['policy_loss'])
        vf_loss.append(metric['vf_loss'])
    
    p = 100
    mean_rewards = np.array([np.mean(rewards[i-p:i+1]) 
                    if i >= p else np.mean(rewards[:i+1]) 
                    for i, _ in enumerate(rewards)])
    std_rewards = np.array([np.std(rewards[i-p:i+1])
                    if i >= p else np.std(rewards[:i+1])
                    for i, _ in enumerate(rewards)])
    
    fig = plt.figure(constrained_layout=True, figsize=(15, 8))
    gs = fig.add_gridspec(2, 4)
    ax0 = fig.add_subplot(gs[:, :-2])
    ax0.fill_between(np.arange(len(mean_rewards)), 
                    mean_rewards - std_rewards, 
                    mean_rewards + std_rewards, 
                    label='Standard Deviation', alpha=0.3)
    ax0.plot(mean_rewards, label='Mean Rewards')
    ax0.set_ylabel('Rewards')
    ax0.set_xlabel('Episode')
    ax0.set_title('Training Rewards')
    ax0.legend()
    
    ax1 = fig.add_subplot(gs[0, 2:])
    ax1.plot(pol_loss)
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_title('Policy Loss')
    
    ax2 = fig.add_subplot(gs[1, 2:])
    ax2.plot(vf_loss)
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_title('Value Function Loss')
    
    plt.show()

