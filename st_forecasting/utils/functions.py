import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm


def pad_sequence(x: np.ndarray, max_size: int) -> np.ndarray:
    if x.shape[0] >= max_size:
        return x[-max_size:]
    else:
        l, f, h, w = x.shape
        padded = np.zeros((max_size, f, h, w))
        padded[-x.shape[0]:] = x
        return padded

    
def to_timeseries(i: int, frames: np.ndarray, max_size: int, horizon: int) -> (np.ndarray, np.ndarray):
    """Lazy implementation of x/y split of input sequence"""

    x = frames[:i]
    return pad_sequence(x[:-horizon], max_size).astype(np.float32), x[-horizon:].astype(np.float32)


def to_df(stream: np.ndarray, timesteps: np.ndarray, ids: np.ndarray) -> pd.DataFrame:
    """
    Converts sequences to pandas.DataFrame

    :param stream: Intput sequence
    :param timesteps: Initial timesteps used
    :param ids: Matrix with squares ids
    """

    df_list = []

    for i, ts in tqdm(enumerate(timesteps), 'Converting to DataFrame...', total=len(timesteps)):
        sub = pd.DataFrame({'square_id': ids, 'intents': stream[i].flatten()})
        sub['ts'] = ts
        df_list.append(sub)

    data = pd.concat(df_list, ignore_index=True)
    return data


def denoise_by_square(data: pd.DataFrame) -> pd.DataFrame:
    denoised_list = []
    ids = np.sort(data['square_id'].unique())

    for sid in tqdm(ids, 'Denoising...'):
        sid_ts = data[data.square_id == sid].sort_values(['ts']).copy()
        sm_sid_ts = signal.savgol_filter(sid_ts['intents'].values, window_length=4, polyorder=1)
        sid_ts['intents'] = sm_sid_ts
        denoised_list.append(sid_ts)

    denoised = pd.concat(denoised_list, ignore_index=True).sort_values(['ts', 'square_id'])

    return denoised


def to_stream(data: pd.DataFrame, squares_df: pd.DataFrame) -> np.ndarray:
    df = data.copy()
    n = int(np.sqrt(len(squares_df)))

    # Quering timestep's index is faster that quering timestep itself (6x speedup)
    tss = df.ts.unique()
    ts_dict = {uts: x for uts, x in zip(tss, range(len(tss)))}
    df['ts'] = df['ts'].map(ts_dict)
    tss = np.sort(df.ts.unique())

    frames = []

    for uts in tqdm(tss, 'Converting to stream...'):
        sub = df[df['ts'] == uts].drop('ts', axis=1)
        sub = sub.sort_values(['square_id'])
        frame = np.expand_dims(np.rot90(sub['intents'].values.reshape(n, n)), axis=0)
        frames.append(frame)

    return np.array(frames)
