from autoop.core.ml.artifact import Artifact
import pandas as pd
import io
from matplotlib import pyplot as plt


class Dataset(Artifact):
    """
    Represents a dataset artifact, inheriting from the Artifact class.
    This class provides functionality to load data from a Pandas DataFrame,
    save it as a CSV, and read it back into a DataFrame.

    Attributes:
        name (str): The name of the dataset.
        asset_path (str): The file path where the dataset is stored.
        version (str): The version of the dataset.
        data (bytes): The raw data of the dataset in CSV format.
        metadata (dict): Additional metadata associated with the dataset.
        type (str): The type of artifact, which is always "dataset".
        tags (list[str]): A list of tags associated with the dataset.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the Dataset instance.

        Args:
            *args: Variable length argument list passed
            to the Artifact constructor.
            **kwargs: Arbitrary keyword arguments passed
            to the Artifact constructor.
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame, name: str, asset_path: str, version: str = "1.0.0"
    ) -> "Dataset":
        """
        Create a Dataset instance from a Pandas DataFrame.

        Args:
            data (pd.DataFrame): The dataset as a Pandas DataFrame.
            name (str): The name of the dataset.
            asset_path (str): The file path where the dataset will be stored.
            version (str, optional): The version of the dataset.
            Defaults to "1.0.0".

        Returns:
            Dataset: The Dataset instance created from the provided DataFrame.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        Read the dataset's raw data and convert it back
        into a Pandas DataFrame.

        Returns:
            pd.DataFrame: The dataset as a Pandas DataFrame.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Save the provided Pandas DataFrame as CSV and return it as bytes.

        Args:
            data (pd.DataFrame): The data to save.

        Returns:
            bytes: The CSV data as bytes.
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)


class DataPlotter:
    """
    A class to plot instances of the Dataset class
    """

    def __init__(
        self,
        data: Dataset,
    ) -> None:
        """
        Initialize the DataPlotter instance.

        Args:
            data (Dataset): The dataset instance to plot.
        """
        self._data = data

    def hist_1d(self, variable: str, **kwargs) -> plt.Figure:
        """
        Plots a 1D histogram of the dataset for the given variable

        Arguments:
        variable (str): The variable to plot
        **kwargs: Arbitrary keyword arguments for plt.hist

        Returns:
        None
        """
        data = self._data[variable]

        if "bins" not in kwargs:
            kwargs["bins"] = 20

        plt.hist(data, **kwargs)
        plt.xlabel(variable)
        plt.ylabel("Frequency")
        plt.title(f"{variable} histogram")
        plt.show()
        return plt.gcf()

    def scatter_2d(self, x_variable: str,
                   y_variable: str, **kwargs) -> plt.Figure:
        """
        Plots a 2D scatter plot of the dataset for the given
        x and y variables.

        Arguments:
        x_variable (str): The x variable to plot
        y_variable (str): The y variable to plot
        **kwargs: Arbitrary keyword arguments for plt.scatter

        Returns:
        None
        """
        x_data = self._data[x_variable]
        y_data = self._data[y_variable]

        plt.scatter(x_data, y_data, **kwargs)
        plt.xlabel(x_variable)
        plt.ylabel(y_variable)
        plt.title(f"{x_variable} vs {y_variable}")
        plt.show()
        return plt.gcf()

    def scatter_3d(
        self, x_variable: str, y_variable: str, z_variable: str, **kwargs
    ) -> plt.Figure:
        """
        Plots a 3D scatter plot of the dataset for the given
        x, y and z variables.

        Arguments:
        x_variable (str): The x variable to plot
        y_variable (str): The y variable to plot
        z_variable (str): The z variable to plot
        **kwargs: Arbitrary keyword arguments for plt.scatter

        Returns:
        None
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        x_data = self._data[x_variable]
        y_data = self._data[y_variable]
        z_data = self._data[z_variable]
        try:
            ax.scatter(x_data, y_data, z_data)
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_zlabel(z_variable)
            plt.title(f"{x_variable} vs {y_variable} vs {z_variable}")
            plt.show()
            return fig
        except ValueError:
            raise ValueError
