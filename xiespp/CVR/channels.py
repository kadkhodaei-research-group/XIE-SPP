import numpy as np
import pandas as pd
from pathlib import Path

# from utility.util_magpie import get_magpie_data
# try:
#     import importlib_resources
#     pt_path = importlib_resources.files('xiespp.CVR') / 'periodic_table.csv'
# except ModuleNotFoundError:
#     pt_path = Path(__file__).resolve().parent / 'periodic_table.csv'
pt_path = Path(__file__).resolve().parent / 'periodic_table.csv'
assert pt_path.exists(), f'Periodic table file not found at {pt_path}'
pt = pd.read_csv(pt_path)


class ImageChannel:
    """
    A class used to represent an Image Channel.
    """

    def __init__(
            self, values, label, normalization_factor=None, atomic_number=None, dtype=None
    ):
        """
        Constructs all the necessary attributes for the image channel object.

        Parameters
        ----------
            values : list
                The values for the image channel.
            label : str
                The label for the image channel.
            normalization_factor : float, optional
                The normalization factor for the image channel (default is None).
            atomic_number : list, optional
                The atomic numbers corresponding to the values (default is a list from 1 to 118 inclusive).
            dtype : str, optional
                The data type for the values (default is None, which lets the system choose the appropriate data type).
        """
        if atomic_number is None:
            atomic_number = list(range(1, 118 + 1))
        values = np.array(values)

        self.label = label
        self.values = values
        self.dtype = self.set_dtype(dtype)
        self.normalization_factor = self.set_normalization_factor(normalization_factor)
        self.values_norm = self.values / self.normalization_factor
        # if self.normalization_factor == 1:
        #     self.values_norm = self.values

        self.df = pd.DataFrame({"atomic_number": atomic_number, label: self.values})
        self.df_norm = pd.DataFrame(
            {"atomic_number": atomic_number, label: self.values_norm}
        )

    def set_dtype(self, dtype):
        """
        Sets the data type for the image channel's values. If no data type is provided,
        the function infers the data type based on the values.

        Parameters
        ----------
        dtype : str, optional
            The desired data type (default is None, which infers the data type from the values).

        Returns
        -------
        str
            The determined data type for the values.

        If all values are integers and fall between 0 and 255 inclusive, the function sets the data type as 'uint8'.
        Otherwise, it sets the data type as 'float32'.
        """
        values = self.values
        if dtype is None:
            if (values == np.floor(values)).all():
                if (values >= 0).all() & (values <= 255).all():
                    dtype = "uint8"
                else:
                    dtype = "float32"
            else:
                dtype = "float32"
        return dtype

    def set_normalization_factor(self, normalization_factor):
        """
        Sets the normalization factor for the image channel's values. If no normalization factor is provided,
        the function determines the normalization factor based on the values and data type.

        Parameters
        ----------
        normalization_factor : float, optional
            The desired normalization factor (default is None, which computes the normalization factor from the values).

        Returns
        -------
        float
            The determined normalization factor for the values.

        If the normalization factor is not provided and the data type is integer,
        the normalization factor is set as the maximum value among the values plus 1.
        Otherwise, it's set as the maximum value among the values.
        """

        values = self.values
        if normalization_factor is None:
            normalization_factor = np.max(values)
            if "int" in self.dtype:
                normalization_factor = np.max(values) + 1
        return normalization_factor

    def __repr__(self):
        return f"Channel: {self.label}"


def channel_setup_group_number(
        lanthanoids_group_number=3.5, actinoids_group_number=3.5
):
    """
    Creates an ImageChannel object based on atomic group numbers for lanthanoids and actinoids.
    It modifies the group numbers for lanthanoids and actinoids in a given periodic table DataFrame
    and uses the resulting group numbers to set up an ImageChannel.

    Parameters
    ----------
    lanthanoids_group_number : float, optional
        The desired group number for lanthanoids (default is 3.5).
    actinoids_group_number : float, optional
        The desired group number for actinoids (default is 3.5).
    """
    label = "group"
    df = pt  # .copy()
    lanthanoids_start = 57
    lanthanoids_end = 71
    actinoids_start = 89
    actinoids_end = 103

    ind = (df["atomic number"] >= lanthanoids_start) & (
            df["atomic number"] <= lanthanoids_end
    )
    df.loc[ind, label] = lanthanoids_group_number

    ind = (df["atomic number"] >= actinoids_start) & (
            df["atomic number"] <= actinoids_end
    )
    df.loc[ind, label] = actinoids_group_number

    values = df[label].to_numpy().astype("uint8")
    channel = ImageChannel(values=values, label=label)

    return channel


def channel_setup_period_number():
    """
    Creates an ImageChannel object based on atomic period numbers.
    It uses the period numbers from a given periodic table DataFrame
    to set up an ImageChannel.
    """
    label = "period"
    df = pt  # .copy()

    values = df[label].to_numpy().astype("uint8")
    channel = ImageChannel(values=values, label=label)

    return channel


def channel_setup_atomic_number():
    """
    Creates an ImageChannel object based on atomic numbers.
    It uses the atomic numbers from a given periodic table DataFrame
    to set up an ImageChannel.
    """
    df = pt  # .copy()

    values = df["atomic number"].to_numpy().astype("uint8")
    channel = ImageChannel(values=values, label="atomic_number")

    return channel


class ChannelsManager:
    """
    A class used to manage multiple ImageChannel objects.
    """
    # Store the value by which channels should be normalized
    # Create image normalizer
    def __init__(self, dtype_norm="float32"):
        # df = pd.read_csv('./periodic_table.csv')
        # or
        df = pt.copy()
        # Setting the index to be the same as atomic number
        df.set_index("atomic number", drop=False, inplace=True)
        df.rename(columns={"atomic number": "atomic_index"}, inplace=True)
        df = df[["symbol", "atomic_index"]]
        self.df = df
        self.df_norm = df.copy()
        self.channels = []
        self.dtype = None
        self.dtype_norm = dtype_norm
        pass

    def add_channel(self, channel: ImageChannel):
        """
        Adds an ImageChannel object to the channels' manager. The method checks if a channel with
        the same label already exists. If not, it appends the new channel to the 'channels' list,
        and adds its values and normalized values to 'df' and 'df_norm' respectively.
        Then it sets the data type for the channels in the manager.
        """
        assert (
                channel.label not in self.df.columns
        ), f"Channel {channel.label} already exists."
        self.channels.append(channel)
        self.df[channel.label] = channel.values
        self.df_norm[channel.label] = np.array(channel.values_norm).astype(
            self.dtype_norm
        )
        self.set_dtype()
        pass

    def add_channels_from_df(self, df, ignore_cols=None):
        """
        Adds channels from a DataFrame to the ChannelsManager. Each column in the DataFrame, except those
        in 'ignore_cols', is treated as a separate channel. The method creates an ImageChannel for each
        column and adds it to the ChannelsManager.
        """
        cols = df.columns.tolist()
        if ignore_cols is None:
            ignore_cols = []
        for c in cols:
            if c in ignore_cols:
                continue
            chn = ImageChannel(df[c].to_numpy(), c)
            self.add_channel(chn)

    def set_dtype(self, dtype=None):
        """
        Sets the data type for the channels in the manager. If the data type is not provided, the method
        """
        if dtype is not None:
            self.dtype = dtype
        else:
            self.find_dtype()
        pass

    def find_dtype(self):
        """
        Finds the data type for the channels in the manager. The method checks the data types of the channels
        """
        dtype = pd.Series(i.dtype for i in self.channels)
        dtype_order = ["float", "float32", "uint8", "uint", "int"]
        for d in dtype_order:
            if pd.Series([d]).isin(dtype)[0]:
                self.dtype = d
                return
        raise Exception("dtype has to be set manually")

    def __repr__(self):
        return (
                f"Channels list [{len(self.channels)}]: ["
                + ", ".join([str(i).replace("Channel: ", "") for i in self.channels])
                + "]"
        )

    def __getitem__(self, item):
        return self.channels[item]


channels_gen = ChannelsManager()
channels_gen.add_channel(channel_setup_atomic_number())
channels_gen.add_channel(channel_setup_group_number())
channels_gen.add_channel(channel_setup_period_number())
# Adding all other physical properties:
# channels_gen.add_channels_from_df(
#     get_magpie_data(), ignore_cols=["symbol", "OxidationStates"]
# )

# channels_gen.add_channel(ImageChannel(values=v, label='label'))
pass
