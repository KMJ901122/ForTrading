import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

'''
Main feautures

Width : number of time stemps of the input and label windows
Time offset between them
Which features are used as inputs, labels, or both

Handle the indexes and offsets as shown in the diagrams above.
Split windows of features into a (features, labels) pairs.
Efficiently generate batches of these windows from the training, evaluation, and test data, using tf.data.Datasets.
Plot the content of the resulting windows.
'''

class WindowGenerator(object):
    def __init__(self, data, input_width, label_width, shift, label_columns=None):
        data=data.dropna()
        print('Data"s shape is', data.shape)
        print('Split data as time and others')

        self.date_time=pd.to_datetime(data.pop('Date'), format='%Y-%m-%d')

        print('date time is ', self.date_time)
        length_of_data=len(data)

        split_nbr=int(length_of_data*0.7)
        split_nbr_2=int(length_of_data*0.8)

        train_df=data[: split_nbr]
        train_mean=train_df.mean()
        train_std=train_df.std()

        # data and time
        self.train_df=(train_df-train_mean)/train_std
        self.val_df=(data[split_nbr: split_nbr_2]-train_mean)/train_std
        self.test_df=(data[split_nbr_2: ]-train_mean)/train_std
        self.train_dt=self.date_time[: split_nbr]
        self.val_df=self.date_time[split_nbr: split_nbr_2]
        self.test_df=self.date_time[split_nbr_2: ]
        self.number_of_features=train_df.shape[-1]
        print('train data"s shape is ', self.train_df.shape)


        # Work out the label column indicces.
        self.label_columns=label_columns
        if label_columns is not None:
            self.label_columns_indices={name: i for i, name in enumerate(label_columns)}
        self.column_indices={name:i for i,name in enumerate(self.train_df.columns)}

        # Work out the window parameters.
        self.input_width=input_width
        self.label_width=label_width
        self.shift=shift

        self.total_window_size=input_width+shift

        self.input_slice=slice(0, input_width)
        self.input_indices=np.arange(self.total_window_size)[self.input_slice]

        self.label_start=self.total_window_size-self.label_width
        self.labels_slice=slice(self.label_start, None)
        self.label_indices=np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs=features[:, self.input_slice, :]
        labels=features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels=tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], aixs=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None):
        plot_col=self.column_indices.keys()
        max_subplots=self.number_of_features
        inputs, labels=self.example
        plt.figure(figsize=(12, 8))
        plot_col_index=self.column_indcies[plot_col]
        max_n=min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_subplots, 1, 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index], label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index=self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index=plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index], edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                          marker='X', edgecolors='k', label='Predictions',
                          c='#ff7f0e', s=64)
            if n==0:
                plt.legend()

            plt.xlabel('Time [h]')

    def make_dataset(self, data):
        data=np.array(data, dtype=np.float32)
        ds=tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=False,
        batch_size=32,
        )

        ds=ds.map(self.split_window)
        return ds
    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_datset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result=getattr(self, '_example', None) # if there exists '_example' meothd then  result=_example <- method, if not, result = None
        if result is None:
            # No example batch was found, so get one from the '.train' dataset
            result=next(iter(self.train))
            # And cach it for next time
            self._example=result
        return result


datadir=r'C:\Users\DELL\Desktop\data\csv\Korea\Samsung.csv'
data=pd.read_csv(datadir)

input_width=6
label_width=1
shift=1

w=WindowGenerator(data, input_width, label_width, shift)
print(w.train.element_spec)
# w.plot()
# inputs, labels=w.split_window(np.array(w.train_df))

# print(inputs, labels)
# print(w.train_df)
# print(w.train_dt)
