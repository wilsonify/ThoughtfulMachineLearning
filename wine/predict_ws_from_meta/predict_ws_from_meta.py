import random

import pandas as pd
from matplotlib import pyplot as plt
from pycaret.regression import get_config
from pycaret.regression import (
    setup,
    compare_models,
    create_model,
    tune_model,
    ensemble_model,
    models,
    blend_models,
    stack_models,
    interpret_model,
    automl,
    predict_model,
    save_model,
    load_model,
    convert_model,
    create_api,
    create_docker
)
from pycaret.utils import version
from yellowbrick.regressor import CooksDistance

id_cols = ['productID', "name", "pProductID", "wine_url", "pageName", "uploadDate"]
target_cols = ['WS']
categorical_cols = [
    "description", "productPrice", "productCompetitiveIntensity", "ProductAvailability", "priceCurrency",
    "additionalType", "productOrigin", "productVarietal", "productRegion"
]
extra_cols = ["shippingRegion", "shipToState"]
other_ratings = ['JS', 'WW', 'D', 'BH', 'W&S', 'WE', 'RP', 'JD', 'SJ', 'V', 'CG', 'TP']
predictors = [
    'productStock',
    'price',
    'prodAlcoholPercent_percent'
    # 'averageRating_bestRating',
    # 'averageRating_worstRating',
    # 'bestRating',
    # 'worstRating',
    # 'prodAlcoholVolume_text',
]

print(f"pycaret version = {version()}")
print("1. Loading Dataset")
df = pd.read_csv("wine_spectator.csv")
df = df.drop_duplicates('wine_url')

# +

unnamed_cols = df.filter(regex="Unnamed")
df = df.drop(categorical_cols, axis=1)
df = df.drop(extra_cols, axis=1)
df = df.drop(unnamed_cols, axis=1)
df = df.drop(other_ratings, axis=1)

df = df[id_cols + predictors + target_cols]
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
df_train = df_train.dropna(subset=target_cols, axis=0)
df_train_id = df_train[id_cols]
df_train = df_train.drop(id_cols, axis=1)
# -

df_test = df.loc[not_picked_indices, :]
df_test_id = df_test[id_cols]
df_test_truth = df_test[target_cols]
df_test = df_test.drop(id_cols, axis=1)
df_test = df_test.drop(target_cols, axis=1)

print(list(df.describe()))

visualizer = CooksDistance()
visualizer.fit(df_train[predictors], df_train["WS"])
# visualizer.show()

df_train['cooks'] = visualizer.distance_
high_influence = df_train['cooks'] > 1.0
high_price = df_train['price'] > 500

df_train = df_train[~high_influence & ~high_price]
df_train = df_train.drop('cooks', axis=1)

df_train_id = df_train_id[~high_influence & ~high_price]

print("2. Initialize Setup")
reg1 = setup(
    data=df_train,
    target="WS",
    experiment_name='ws_from_meta',
    # imputation_type='iterative',
)

# +
print("3. Compare Baseline")

#best_model = compare_models(fold=5)
# -

#print("4. Create Model")
#lr = create_model('lr')
#lr = tune_model(lr, n_iter=5, optimize='RMSE')
# evaluate_model(lr)

#rf = create_model('rf')
#rf = tune_model(rf, n_iter=5, optimize='RMSE')
# evaluate_model(rf)

#knn = create_model('knn')
#knn = tune_model(knn, n_iter=5, optimize='RMSE')
# evaluate_model(knn)

#print("5. Tune Hyperparameters")
#lightgbm = create_model('lightgbm')
#lightgbm = tune_model(lightgbm, n_iter=5, optimize='MAE')
# evaluate_model(lightgbm)

print("6. Ensemble Model")

dt = create_model('dt')

bagged_dt = ensemble_model(dt, n_estimators=50)

boosted_dt = ensemble_model(dt, method='Boosting')

print("7. Blend Models")

list(models().index)

top_five = compare_models(n_select=5, fold=5, include=list(models().index))

blender = blend_models(estimator_list=top_five)

print("8. Stack Models")

stacker = stack_models(estimator_list=top_five)

#print("9. Analyze Model")

# plot_model(dt)
# plot_model(dt, plot='error')
# plot_model(dt, plot='feature')
# evaluate_model(dt)

#print("10. Interpret Model")

#interpret_model(lightgbm)

#interpret_model(lightgbm, plot='correlation')

#interpret_model(lightgbm, plot='reason', observation=12)

print("11. AutoML()")

best = automl(optimize='RMSE')
print(best)

#print("12. Predict Model")

#pred_holdouts = predict_model(lightgbm)
#pred_holdouts.head()

predict_new = predict_model(best, data=df_test)
predict_new.head()

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(predict_new['prediction_label'], df_test_truth)
ax.set_xlabel('predicted WS rating')
ax.set_ylabel('actual WS rating')

save_model(best, model_name='best-model')

#loaded_bestmodel = load_model('best-model')
#print(loaded_bestmodel)

#print(loaded_bestmodel[0])

#X_train = get_config('X_train')
#X_train.head()

#convert_model(best_model, 'c')
#convert_model(best_model, 'python')
#convert_model(best_model, 'java')
#convert_model(best_model, 'javascript')
#convert_model(best_model, 'c')
#convert_model(best_model, 'c#')
#convert_model(best_model, 'f#')
#convert_model(best_model, 'go')
#convert_model(best_model, 'haskell')
#convert_model(best_model, 'php')
#convert_model(best_model, 'powershell')
#convert_model(best_model, 'r')
#convert_model(best_model, 'ruby')
#convert_model(best_model, 'vb')
#convert_model(best_model, 'dart')

create_api(best, api_name='predict_ws_from_meta_api')

create_docker(api_name='predict_ws_from_meta')
