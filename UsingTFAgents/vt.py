from __future__ import absolute_import
import pandas as pd
import seaborn as sns

class plots(object):
    def __init__(self, df):
        self.df=df

    def v_plot(self, mean=None, std=None):
        df=self.df
        print(type(self.df))
        print('Check whether type of data is pandas.DataFrame or not')

        if mean==None or std==None:
            print('initial mean or std is none')
            mean=self.df.mean()
            std=self.df.std()
            print('mean is ', mean)
            print('std is', std)

        df=(df-mean)/std
        df=df.melt(var_name='Column', value_name='Normalized')
        import matplotlib.pyplot as plt
        # import seaborn as sns

        plt.figure(figsize=(12, 6))
        # 아래가 문제임
        return df
        # ax=(self.sns).violinplot(x='Column', y='Normalized', data=df)
        # _ = ax.set_xticklabels(df.keys(), rotation=90)
        # print(df.columns)
        # print('plt.show()')
        # plt.show()


datadir=r'C:\Users\DELL\Desktop\data\csv\Korea\Samsung.csv'
data=pd.read_csv(datadir)
v_tools=plots(data)
df=v_tools.v_plot()
# print(df.columns)
ax=sns.violinplot(x=df.columns[0], y=df.columns[1], data=df)
_ = ax.set_xticklabels(df.keys(), rotation=90)
plt.show()
# v_plot(data)
