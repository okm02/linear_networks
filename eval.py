import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def extract_metrics(path: str) -> pd.DataFrame:
    """
    :param path: path to metric logs (csv format by pytorch lightning)
    :returns: metrics tracked when training
    """
    df = pd.read_csv(path)

    # extract first training metrics
    train_metrics = df[df['val_loss'].isna()]
    # keep maximum step metrics
    idx = train_metrics.groupby('epoch')['step'].idxmax()
    idx.reset_index(drop=True, inplace=True)
    train_metrics = train_metrics.loc[idx]
    train_metrics.drop(columns=['step', 'val_loss'], inplace=True)
    train_metrics.reset_index()

    # extract second validation metrics
    val_metrics = df[~df['val_loss'].isna()]
    val_metrics = val_metrics.drop(columns=['train_loss', 'step'])
    val_metrics.reset_index()

    metric_df = pd.merge(train_metrics, val_metrics, on='epoch', how='inner')
    return metric_df


def plot_metrics(metric_df: pd.DataFrame, plot_title:str, plt_name: str):
    """
    :param metric_df  : metrics tracked when training
    :param plot_title : title of plot with training metrics
    :param plt_name   : name of the plot to be saved
    """

    fig = plt.Figure(figsize=(12, 8))
    plt.plot(metric_df['epoch'], metric_df['train_loss'], label='train')
    plt.plot(metric_df['epoch'], metric_df['val_loss'], label='val')
    plt.grid()
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction Loss')
    plt.title(plot_title)
    plt.savefig(f'plots/{plt_name}.png')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mdl_path', type=str, help='path of having csv logs of metrics')
    parser.add_argument('--plot_title', type=str, help='title of plot')
    parser.add_argument('--plot_name', type=str, help='name of file to save the plot to')
    args = parser.parse_args()
    
    plot_metrics(metric_df=extract_metrics(args.mdl_path), plot_title=args.plot_title, plt_name=args.plot_name)
