import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

from tml_wine.predict_ws_inference import composite

if __name__ == "__main__":
    composite(DataFrame(dict(price=[25.00], ws_pred=[95.0])), visualize=True)
    # Generate combinations of price and rating
    price_values = np.linspace(1, 99, 20)
    rating_values = np.linspace(1, 99, 20)
    combinations = [(p, r) for p in price_values for r in rating_values]
    df = pd.DataFrame(combinations, columns=['price', 'ws_pred'])
    zs = composite(df)
    df['score'] = zs
    X = df['price'].values.reshape(price_values.shape[0], -1)
    Y = df['ws_pred'].values.reshape(rating_values.shape[0], -1)
    Z = zs.reshape(X.shape)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.8, cmap='viridis')
    ax.view_init(30, 200)
    ax.set_xlabel('price')
    ax.set_ylabel('rating')
    ax.set_zlabel('score')
    plt.show()
