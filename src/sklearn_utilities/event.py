import numpy as np
import pandas as pd
from pandas import DatetimeIndex
from pandas.core.series import Series


def since_event(event: "Series[bool]") -> "Series[float]":
    """Calculate the elapsed time since the last event.

    Parameters
    ----------
    event : Series[bool]
        Whether the event occurred.

    Returns
    -------
    Series[float]
        The difference between the index of the last event
        and the current index.
        If the index is a DatetimeIndex, the unit is hours.
    """
    event = event.astype(bool, copy=True)
    orig_idx = event.index
    idx = event.index
    if isinstance(idx, DatetimeIndex):
        # without converting to float, calculation will be extremely slow
        idx = pd.to_numeric(idx, downcast="float") / 1e9 / 60 / 60
        event.index = idx
    last_event_index = Series(np.nan, index=idx)
    last_event_index[event] = idx
    last_event_index.ffill(inplace=True)
    res = idx - last_event_index
    res.index = orig_idx
    return res


def until_event(event: "Series[bool]") -> "Series[float]":
    """Calculate the elapsed time until the next event.

    Parameters
    ----------
    event : Series[bool]
        Whether the event occurred.

    Returns
    -------
    Series[float]
        The difference between the index of the next event
        and the current index.
        If the index is a DatetimeIndex, the unit is hours.
    """
    event = event.astype(bool, copy=True)
    orig_idx = event.index
    idx = event.index
    if isinstance(event.index, DatetimeIndex):
        # without converting to float, calculation will be extremely slow
        idx = pd.to_numeric(idx, downcast="float") / 1e9 / 60 / 60
        event.index = idx
    next_event_index = Series(np.nan, index=idx)
    next_event_index[event] = idx
    next_event_index.bfill(inplace=True)
    res = next_event_index - idx
    res.index = orig_idx
    return res
