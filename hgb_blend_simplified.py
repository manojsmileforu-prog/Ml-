import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error

TRAIN, TEST = "train.csv", "test.csv"

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)


def add_features(df):
    df = df.copy()
    df["route"] = df.source_city + "_" + df.destination_city
    df["fnum"] = df.flight.str.extract(r"(\d+)").astype(float)
    df["stops_n"] = df.stops.map({"zero": 0, "one": 1, "two_or_more": 2})
    df["dep_arr"] = df.departure_time + "_" + df.arrival_time
    df["air_class"] = df.airline + "_" + df.flight_class
    df["air_route"] = df.airline + "_" + df.route
    df["cls_dl"] = (df.flight_class == "Business").astype(int) * 1000 + df.days_left
    df["dl_bucket"] = pd.cut(
        df.days_left, [-1, 1, 2, 3, 5, 7, 14, 21, 30, 60], labels=False
    )
    df["logdur"] = np.log1p(df.duration)
    return df


all_data = add_features(pd.concat([train.drop(columns="price"), test]))
categorical = [
    "airline", "source_city", "departure_time", "stops", "arrival_time",
    "destination_city", "flight_class", "route", "dep_arr", "air_class", "air_route"
]
all_data[categorical] = all_data[categorical].astype("category")

features = [c for c in all_data.columns if c not in ("id", "flight")]
X = all_data.iloc[:len(train)][features]
X_test = all_data.iloc[len(train):][features]

# Extra features use train and test data, without using prices.
extra = all_data.copy()
extra["flight_count"] = all_data.groupby("flight")["id"].transform("count")
flight_class = all_data.flight.astype(str) + all_data.flight_class.astype(str)
extra["flight_cls_count"] = flight_class.map(flight_class.value_counts())
extra["dur_leg"] = extra.duration / (extra.stops_n + 1)

extra_features = [c for c in extra.columns if c not in ("id", "flight")]
X_extra = extra.iloc[:len(train)][extra_features]
X_test_extra = extra.iloc[len(train):][extra_features]

y = train.price.values
base_params = {
    "max_iter": 1500,
    "learning_rate": 0.05,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 10,
    "l2_regularization": 1.0,
    "categorical_features": "from_dtype",
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 50,
    "random_state": 0,
}

# Each model stores: parameters, whether to log-transform prices, train features, test features.
models = {
    "log (your model)": (base_params, True, X, X_test),
    "raw price": ({**base_params, "random_state": 1}, False, X, X_test),
    "log + extra features": ({**base_params, "random_state": 2}, True, X_extra, X_test_extra),
    "log, 127 leaves": (
        {**base_params, "max_leaf_nodes": 127, "min_samples_leaf": 5, "random_state": 4},
        True, X, X_test,
    ),
    "log, 70% columns": ({**base_params, "max_features": 0.7, "random_state": 5}, True, X, X_test),
    "log, seed 6": ({**base_params, "random_state": 6}, True, X, X_test),
}

folds = list(KFold(5, shuffle=True, random_state=1).split(X))
oof_predictions, test_predictions = {}, {}

for name, (params, use_log, train_X, test_X) in models.items():
    target = np.log1p(y) if use_log else y
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test_X))

    for train_idx, valid_idx in folds:
        model = HistGradientBoostingRegressor(**params).fit(
            train_X.iloc[train_idx], target[train_idx]
        )
        valid_pred = model.predict(train_X.iloc[valid_idx])
        fold_test_pred = model.predict(test_X)

        if use_log:
            valid_pred = np.expm1(valid_pred)
            fold_test_pred = np.expm1(fold_test_pred)

        oof[valid_idx] = valid_pred
        test_pred += fold_test_pred / len(folds)

    oof_predictions[name] = np.clip(oof, 0, None)
    test_predictions[name] = np.clip(test_pred, 0, None)
    print(
        f"{name:24s} R2={r2_score(y, oof_predictions[name]):.4f}  "
        f"MAE={mean_absolute_error(y, oof_predictions[name]):.2f}"
    )

# Blend the models using non-negative weights.
names = list(models)
oof_matrix = np.column_stack([oof_predictions[name] for name in names])
test_matrix = np.column_stack([test_predictions[name] for name in names])

weights, _ = nnls(oof_matrix, y)
weights /= weights.sum()

print("\nBlend weights:", dict(zip(names, weights.round(3))))
print("Simple average  R2:", round(r2_score(y, oof_matrix.mean(axis=1)), 4))
blended_oof = oof_matrix @ weights
print(
    "Weighted blend  R2:", round(r2_score(y, blended_oof), 4),
    " MAE:", round(mean_absolute_error(y, blended_oof), 2),
)

submission = pd.DataFrame({
    "id": test.id,
    "price": np.round(test_matrix @ weights, 2),
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv", submission.shape)
