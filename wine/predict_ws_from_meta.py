import random

import numpy as np
import pandas as pd
from pycaret.regression import get_config
from pycaret.regression import set_config
from pycaret.regression import (
    setup,
    compare_models,
    create_model,
    tune_model,
    ensemble_model,
    models,
    blend_models,
    stack_models,
    plot_model,
    evaluate_model,
    interpret_model,
    automl,
    predict_model,
    save_model,
    load_model
)
from pycaret.utils import version

id_cols = ['productID']
target_cols = ['WS']
predictors = [
    'productStock',
    'price',
    'averageRating_bestRating',
    'averageRating_worstRating',
    'bestRating',
    'worstRating',
    'prodAlcoholVolume_text',
    'prodAlcoholPercent_percent',
    'JS', 'WW', 'D', 'BH',
    'W&S', 'WE', 'RP', 'JD',
    'SJ', 'V', 'CG', 'TP'
]

print(f"pycaret version = {version()}")
print("1. Loading Dataset")
df = pd.read_csv("wine_spectator.csv")
indices = list(df.index)
n_samples = len(indices)
train_samples = int(0.8 * n_samples)
test_samples = n_samples - train_samples
print(f"train_samples = {train_samples}")
print(f"test_samples = {test_samples}")
random.shuffle(indices)
pick_k_random_indices = random.sample(indices, train_samples)
not_picked_indices = set(indices).difference(pick_k_random_indices)
df_train = df.loc[pick_k_random_indices, :]
df_train_id = df_train[id_cols]
df_train = df_train.drop(id_cols, axis=1)
df_train = df_train.dropna(subset=target_cols, axis=0)

df_test = df.loc[not_picked_indices, :]
df_test_id = df_test[id_cols]
df_test_truth = df_test[target_cols]
df_test = df_test.drop(id_cols, axis=1)
df_test = df_test.drop(target_cols, axis=1)

print(list(df.describe()))

print("2. Initialize Setup")
reg1 = setup(
    data=df_train,
    target="WS",
    experiment_name='ws_from_meta',
    imputation_type='iterative'
)

print("3. Compare Baseline")

best_model = compare_models(fold=5)

print("4. Create Model")
lightgbm = create_model('lightgbm')
lgbms = [create_model('lightgbm', learning_rate=i) for i in np.arange(0.1, 1, 0.1)]
print(len(lgbms))

print("5. Tune Hyperparameters")
tuned_lightgbm = tune_model(lightgbm, n_iter=50, optimize='MAE')

print(tuned_lightgbm)

print("6. Ensemble Model")

dt = create_model('dt')

bagged_dt = ensemble_model(dt, n_estimators=50)

boosted_dt = ensemble_model(dt, method='Boosting')

print("7. Blend Models")

top_five = compare_models(n_select=5, fold=5, include=list(models().index))

blender = blend_models(estimator_list=top_five)

print("8. Stack Models")

stacker = stack_models(estimator_list=top_five)

print("9. Analyze Model")

plot_model(dt)

plot_model(dt, plot='error')

plot_model(dt, plot='feature')

evaluate_model(dt)

print("10. Interpret Model")

interpret_model(lightgbm)

interpret_model(lightgbm, plot='correlation')

interpret_model(lightgbm, plot='reason', observation=12)

print("11. AutoML()")

best = automl(optimize='MAE')
print(best)

print("12. Predict Model")

pred_holdouts = predict_model(lightgbm)
pred_holdouts.head()

predict_new = predict_model(best, data=df_test)
predict_new.head()

save_model(best, model_name='best-model')

loaded_bestmodel = load_model('best-model')
print(loaded_bestmodel)

set_config(display='diagram')
print(loaded_bestmodel[0])

X_train = get_config('X_train')
X_train.head()
