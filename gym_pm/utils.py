import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def load_data(Type='PdM2'):

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
        df = df[~(df.age + df.ttf <= 10)]
        df.reset_index(drop = True, inplace = True)

    elif Type == 'PdM2':

        df['Failure'] = df.Failure_today.apply(lambda x: 0 if x == 'No' else 1)
        df.Date = pd.to_datetime(df.Date)
        df = df[df.Date > '2016-06']
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
        df = df[~(df.age + df.ttf <= 10)]
        df.reset_index(drop = True, inplace = True)

    return df


def evaluate_baseline(eval_env, duration=100, 
                      repair_policy=0, repair_interval=10, 
                      display=False):
    
    """
        repair_policy
        
        0: Repair when failed
        1: Repair at repair_interval
    """
    
    total_reward = []
    obs = eval_env.reset()
    
    if repair_policy == 0:
        
        for i in range(duration):

            if obs['Failure'][0] == 1:
                action = 0 # Repair
            else:
                action = 1

            obs, reward, done, info = eval_env.step(action)
            total_reward.append(reward)

            if display:
                eval_env.render()
                
    elif repair_policy == 1:
        
        for i in range(duration):

            if (obs['Failure'][0] == 1) or obs['age'][0] >= repair_interval:
                action = 0 # Repair
            else:
                action = 1

            obs, reward, done, info = eval_env.step(action)
            total_reward.append(reward)

            if display:
                eval_env.render()
        
    else:
        
        return "Invalid Policy"
        
    return np.sum(total_reward)


def evaluate_policy(eval_env, trainer,
                    duration=100, RNN=False,
                    display=False):
    
    obs = eval_env.reset()
    df_result = pd.DataFrame(obs)
    action_list = []
    reward_list = []
    
    if RNN == False:

        for i in range(duration):

            action = trainer.compute_action(obs)
            obs, reward, done, info = eval_env.step(action)
            
            df_result = df_result.append(pd.DataFrame(obs), ignore_index=True)
            reward_list.append(reward)
            action_list.append(action)

            if display:
                eval_env.render()
                
    else:
        
        # If use LSTM
        state = trainer.get_policy().model.get_initial_state()

        for i in range(duration):

            action, state, logit = trainer.compute_action(obs, prev_action=1.0,
                                                          prev_reward=0.0, state=state)
            obs, reward, done, info = eval_env.step(action)
            
            df_result = df_result.append(pd.DataFrame(obs), ignore_index=True)
            reward_list.append(reward)
            action_list.append(action)
            
            if display:
                eval_env.render()
    
    df_result = df_result[:-1]
    df_result['action'] = action_list
    df_result['reward'] = reward_list
    
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

