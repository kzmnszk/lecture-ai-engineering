import os
import mlflow
import mlflow.sklearn
import mlflow.tracking
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from mlflow.models.signature import infer_signature
from datetime import datetime

# データ準備
def prepare_data(test_size=0.2, random_state=42):
    # Titanicデータセットの読み込み
    path = "演習1/data/Titanic.csv"
    data = pd.read_csv(path)

    # 必要な特徴量の選択と前処理
    data = data[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
    data["Sex"] = LabelEncoder().fit_transform(data["Sex"])  # 性別を数値に変換

    # 整数型の列を浮動小数点型に変換
    data["Pclass"] = data["Pclass"].astype(float)
    data["Sex"] = data["Sex"].astype(float)
    data["Age"] = data["Age"].astype(float)
    data["Fare"] = data["Fare"].astype(float)
    data["Survived"] = data["Survived"].astype(float)

    X = data[["Pclass", "Sex", "Age", "Fare"]]
    y = data["Survived"]

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# 学習と評価
def train_and_evaluate(
    X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None, random_state=42
):
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy


# モデル保存
def log_model(model, accuracy, duration, params):
    with mlflow.start_run():
        # パラメータをログ
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # メトリクスをログ
        mlflow.log_metric("accuracy", accuracy)

        # 訓練時間をログ
        mlflow.log_param("train_duration_sec", duration)
        # モデルのシグネチャを推論
        signature = infer_signature(X_train, model.predict(X_train))

        # モデルを保存
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_test.iloc[:5],  # 入力例を指定
        )
        # accurecyとparmsは改行して表示
        print(f"***モデルのログ***")
        print(f"accuracy: {accuracy}")
        print(f"params: {params}")
        print(f"duration: {duration} sec")


# メイン処理
if __name__ == "__main__":
    # ランダム要素の設定
    test_size = round(
        random.uniform(0.1, 0.3), 2
    )  # 10%〜30%の範囲でテストサイズをランダム化
    data_random_state = random.randint(1, 100)
    model_random_state = random.randint(1, 100)
    n_estimators = random.randint(50, 200)
    max_depth = random.choice([None, 3, 5, 10, 15])

    # パラメータ辞書の作成
    params = {
        "test_size": test_size,
        "data_random_state": data_random_state,
        "model_random_state": model_random_state,
        "n_estimators": n_estimators,
        "max_depth": "None" if max_depth is None else max_depth,
    }

    # データ準備
    X_train, X_test, y_train, y_test = prepare_data(
        test_size=test_size, random_state=data_random_state
    )

    time_start = datetime.now()
    # 学習と評価
    model, accuracy = train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=model_random_state,
    )
    time_end = datetime.now()
    duration = (time_end - time_start).total_seconds()
    # print(time_start)
    # print(time_end)
    # print(duration)

    # モデル保存
    log_model(model, accuracy, duration, params)

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"titanic_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"モデルを {model_path} に保存しました")

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Default")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                              order_by=['metrics.accuracy DESC'],
                              max_results=5)
    print("=== 過去の精度一覧 ===")
    for run in runs:
        acc = run.data.metrics.get("accuracy")
        print(f"Run ID: {run.info.run_id}, Accuracy: {acc}")