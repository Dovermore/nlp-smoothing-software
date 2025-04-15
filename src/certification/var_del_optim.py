"""
AdaptDel authors authored this file with the help of ChatGPT
"""
import json
import os
import pandas as pd
import numpy as np

from .var_del_mech import VarDelMech


class BinnedLenAgent:
    def __init__(self, bins, values=None):
        """
        Initialize the BinnedLenDel object.

        Parameters:
        bins (list of floats): A list of bin boundaries excluding 0 and infinity.
                               Example: [50, 100] represents bins [0, 50), [50, 100), and [100, ∞).
        values (list, int, or None): A list of values corresponding to each bin,
                                     a single integer (constant for all bins),
                                     or None (returns a random value between 0.7 and 1).
        """
        # Insert 0 and float('inf') implicitly to bins
        self.bins = [0] + bins + [float('inf')]

        # Handle values based on type
        num_bins = len(self.bins) - 1
        if values is None:
            self.values = None  # Special case: random value between 0.7 and 1
        elif isinstance(values, (int, float)):
            self.values = [values] * num_bins  # All bins map to the same numerical value
        elif isinstance(values, list):
            if len(values) != num_bins:
                raise ValueError(
                    f"Number of values ({len(values)}) must match the number of bins ({num_bins})."
                )
            if not all(isinstance(v, (int, float)) for v in values):
                raise TypeError("All elements in the values list must be integers or floats.")
            self.values = values
        else:
            raise TypeError("Values must be a list of numbers (int/float), a single number, or None.")

    def __call__(self, value):
        """
        Look up the bin enclosing the value and return the corresponding mapped value (1 - map_value / value).

        Parameters:
        value (float): The input value to check.

        Returns:
        The mapped value corresponding to the bin enclosing the input, or a random value if values are None.
        """
        if self.values is None:
            # Return a random value between 0.7 and 0.99
            return np.random.uniform(0.7, 0.99)

        for i in range(len(self.bins) - 1):
            if self.bins[i] <= value < self.bins[i + 1]:
                return 1 - self.values[i] / value
        # Handle edge case where value is exactly infinity
        return min(max(1 - self.values[-1] / value, 0), 1)

    def save(self, filepath):
        """
        Save the bins and values to a JSON file.

        Parameters:
        filepath (str): The file path to save the JSON data.
        """
        # Save only the bins excluding 0 and infinity (user-provided bins)
        user_bins = self.bins[1:-1]
        with open(filepath, "w") as f:
            json.dump({"bins": user_bins, "values": self.values}, f)

    def load(self, filepath):
        """
        Load the bins and values from a JSON file and update the current instance.

        Parameters:
        filepath (str): The file path to load the JSON data.
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        self.bins = [0] + data["bins"] + [float('inf')]
        self.values = data["values"]


class VarDelOptim(VarDelMech):
    def __init__(
        self,
        bins,
        values=None,
        agent_dir=None,
        agent_type="len",
        **kwargs,
    ) -> None:
        if agent_type == "len":
            f_del = BinnedLenAgent(
                bins=bins,
                values=values,
            )
        else:
            raise ValueError("Agent type not implemented")
        super().__init__(f_del=f_del, **kwargs)
        self.agent_dir = agent_dir
        self.load_agent(raise_error_not_found=False)

    def save_agent(self, version="best"):
        if self.agent_dir is None:
            return
        agent_path = os.path.join(self.agent_dir, f"{version}.pickle")
        self.f_del.save(agent_path)

    def load_agent(self, version="best", raise_error_not_found=True):
        if self.agent_dir is None:
            return
        agent_path = os.path.join(self.agent_dir, f"{version}.pickle")
        if os.path.exists(agent_path):
            self.f_del.load(agent_path)
        elif raise_error_not_found:
            assert False, "agent file not found"

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(var_del_optim,)"
        )
