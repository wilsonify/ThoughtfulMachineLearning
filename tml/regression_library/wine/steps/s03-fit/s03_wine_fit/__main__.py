from pprint import pprint

from pycaret.regression import (
    load_model
)
from pycaret.regression import setup, create_model, deploy_model, tune_model, save_model

from io_library.read_from_s3 import read_csv_from_s3
from io_library.skl_to_py import save_skl_to_json_fs
from io_library.skl_to_py import save_skl_to_json_s3
from s03_wine_fit import (
    INPUT_BUCKET,
    INPUT_PREFIX,
    OUTPUT_BUCKET,
    OUTPUT_PREFIX,
    X_COLUMNS,
    Y_COLUMNS
)


def happy_path(event):
    df = read_csv_from_s3(bucket=INPUT_BUCKET, key=f"{INPUT_PREFIX}/csv/train.csv")
    df = df[X_COLUMNS + Y_COLUMNS]
    setup(data=df, target=Y_COLUMNS[0], imputation_type="simple")
    rf = create_model("rf")
    rf = tune_model(rf, optimize='MAE', n_iter=50)
    deploy_model(model=rf, model_name="ws_rf", authentication=dict(
        bucket=OUTPUT_BUCKET,
        path=f"{OUTPUT_PREFIX}/ws_rf",
        platform="aws"
    ))

    save_model(rf, model_name='ws_rf')
    loaded_model = load_model('ws_rf')
    save_skl_to_json_s3(loaded_model, bucket=OUTPUT_BUCKET, key=f"{OUTPUT_PREFIX}/ws_rf/rf.json")
    save_skl_to_json_fs(loaded_model, filename="ws_rf.json")


def lambda_handler(event, context):
    print("event")
    pprint(event)

    print("context")
    pprint(context)

    print("start main")
    happy_path(event)
    print("done main")

    response = {"statusCode": 200, "body": "Success from s02_create_wine_dataset lambda"}
    return response


if __name__ == "__main__":
    happy_path({})
