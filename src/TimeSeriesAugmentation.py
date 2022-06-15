import numpy as np
import pandas as pd


class TimeseriesAugmentation:

    """
    This class is use window slicing and window warping for data augmentation.
    NB: The input dataframe should be normilized with values between [-1,1].
    timesteps should be rows and the time series (cycle_id) should be columns
    -------

    Parameters:
    -------

        -reduce_ratio: float (Default = 0.9)
            The percentage of the data we want to keep using window slicing).

        -target_len: int (Default = None)
            The  lenght of the data we want to keep; it can be computed using
            the reduce ratio(It is used for the window slicing)

        - window_ratio: float (Default = 0.1)
            (It is used for the window warping)

        -scales: list  (Default = [0.5, 2.])
                    (It is used for the window warping)

        -out_size int (Default = 2 )
            It define the size of the output.


    Methods
    -------
        window_slice: Take a Dataframe as parameter and return a dataframe,
        (with the desired output size) using the window slicing method.

        window_warp: Take a Dataframe as parameter and return a dataframe,
        (with the desired output size) using the window warping method.

    Example:
    -------
        # Import the required libraries

        import data_augmentation as da
        import pandas as pd
        import numpy as np
        from sklearn import preprocessing

        # create or import your timeseries data

        df = pd.DataFrame(np.random.randn(100,5), columns=list('12345'))
        df

        # scale your data between -1 and 1

        x = df.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)

        #create an instance of the Timeseriesaugmentation class

        Obj1 =da.TimeseriesAugmentation()

        # call the window warping or window slicing method

        d_slice=Obj1.window_slice(df)
        d_slice

        d_warp=Obj1.window_warp(df)
        d_warp

        #NB: in this example we are using all defaults parameter.
        you can plays around with different values

    """

    def __init__(self, out_size: int = 2) -> None:

        assert out_size != 0, "The out size should not be zero"
        self.out_size = out_size

    def window_slice(self,
                     df: pd.DataFrame,
                     reduce_ratio: float = None,
                     target_len: float = None
                     ) -> pd.DataFrame:

        """
        The general concept behind slicing is that the data is augmented
        by slicing time steps off the ends of the pattern
        """

        df = df.T
        df.columns = range(df.shape[1])

        x = df.to_numpy().reshape(df.shape[0], df.shape[1], 1)

        if target_len is None:
            if reduce_ratio is None:
                reduce_ratio = 0.9
                target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
            else:
                target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
        else:
            if reduce_ratio is None:
                target_len = np.ceil(target_len)
            else:
                reduce_ratio = None
                target_len = np.ceil(target_len)

        if target_len >= x.shape[1]:
            return x

        list_Datarame = [df]

        for k in range(self.out_size-1):

            starts = np.random.randint(low=0,
                                       high=x.shape[1]-target_len,
                                       size=(x.shape[0])
                                       ).astype(int)

            ends = (target_len + starts).astype(int)

            ret = np.zeros_like(x)
            for i, pat in enumerate(x):
                for dim in range(x.shape[2]):
                    ret[i, :, dim] = np.interp(
                                               np.linspace(0,
                                                           target_len,
                                                           num=x.shape[1]),
                                               np.arange(target_len),
                                               pat[starts[i]:ends[i], dim]
                                               ).T

            list_Datarame.append(pd.DataFrame(ret.reshape(x.shape[0],
                                 x.shape[1]))
                                 )

        return pd.concat(list_Datarame, axis=0).T

    def window_warp(self,
                    df: pd.DataFrame,
                    window_ratio: float = 0.1,
                    scales: list = [0.5, 2.]
                    ) -> pd.DataFrame:

        """
        Window warping takes a random window of the time series and stretches
        it by 2 or contracts it by 1. While the multipliers are fixed to 1/2
        and 2, they can be modified or optimized to other values.
        """

        df = df.T
        df.columns = range(df.shape[1])

        list_Datarame = [df]

        x = df.to_numpy().reshape(df.shape[0], df.shape[1], 1)

        for k in range(self.out_size-1):

            warp_scales = np.random.choice(scales, x.shape[0])
            warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
            window_steps = np.arange(warp_size)

            window_starts = np.random.randint(low=1,
                                              high=x.shape[1]-warp_size-1,
                                              size=(x.shape[0])
                                              ).astype(int)
            window_ends = (window_starts + warp_size).astype(int)

            ret = np.zeros_like(x)
            for i, pat in enumerate(x):
                for dim in range(x.shape[2]):
                    start_seg = pat[:window_starts[i], dim]
                    window_seg = np.interp(np.linspace(0, warp_size-1,
                                           num=int(warp_size*warp_scales[i])),
                                           window_steps,
                                           pat[window_starts[i]:window_ends[i], dim]
                                           )
                    end_seg = pat[window_ends[i]:, dim]
                    warped = np.concatenate((start_seg, window_seg, end_seg))
                    ret[i, :, dim] = np.interp(np.arange(x.shape[1]),
                                               np.linspace(0, x.shape[1]-1., num=warped.size),
                                               warped
                                               ).T

            list_Datarame.append(pd.DataFrame(ret.reshape(x.shape[0],
                                 x.shape[1]))
                                 )

        return pd.concat(list_Datarame, axis=0).T
