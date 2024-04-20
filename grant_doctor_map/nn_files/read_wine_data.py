import pandas as pd


def read():
    path = 'data/winequality-white.csv'
    df = pd.read_csv(path, delimiter=';')
    return df


if __name__ == '__main__':
    read()