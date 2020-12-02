import numpy as np
import pandas as pd

def RewardDiscount(rewards, gamma=0.99): # weights on later rewards ex (.99)^10 R_0 + 0.99^9 R_1 +... + R_10 = reward
    reward_length=len(rewards)
    discount=np.array([gamma**i for i in range(reward_length)])
    discount.sort()
    return np.dot(rewards, discount)

def StatesPreprocess(datadir, window=10, reward_length=1, gamma=0.99, splits=0.8):
    data=pd.read_csv(datadir)
    closed_data=data['Adj Close'].dropna()
    closed_data=np.array(closed_data) # Adj close prices
    pre_states=np.divide(np.diff(closed_data), closed_data[:-1]) # (p_n-p_{n-1})/p_{n-1}
    split_nbr=int(len(pre_states)*0.8)

    train_pre_states=pre_states[:split_nbr]
    eval_pre_states=pre_states[split_nbr:]
    train_states=[]
    eval_states=[]

    for i in range(window, len(train_pre_states)):
        train_states.append(train_pre_states[i-window: i])

    for i in range(window, len(eval_pre_states)):
        eval_states.append(eval_pre_states[i-window: i])

    train_states=np.array(train_states)
    eval_states=np.array(eval_states)

    discount=np.array([gamma**i for i in range(reward_length)]) # weights more on later rewards
    discount.sort()

    train_rewards=np.array([RewardDiscount(train_pre_states[i: ], gamma) for i in range(len(train_pre_states[window:]))])
    eval_rewards=eval_pre_states[window: ]

    train_rewards=train_rewards.reshape((len(train_rewards), 1))
    eval_rewards=eval_rewards.reshape((len(eval_rewards), 1))

    train_states=np.concatenate((train_states, train_rewards), axis=1)
    eval_states=np.concatenate((eval_states, eval_rewards), axis=1)

    print('In a train step, States length: ', len(train_states))
    print('In a evaluation step, States length: ', len(eval_states))
    print('Window size: ', window)
    print('Reward length: ', reward_length)
    print('Gamma: ', gamma)

    return train_states, eval_states
