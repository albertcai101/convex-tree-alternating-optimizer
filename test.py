from optimizer import BlitzOptimizer
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import dataloader as loader


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, X_valid, y_valid = loader.load_bace()
    D = X_train.shape[1]

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)

    # XGBoost
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    print(f"XGBoost Test Accuracy: {np.mean(y_test_pred == y_test)}")
    print(f"XGBoost Test AUC: {roc_auc_score(y_test,y_test_pred)}")

    y_valid_pred = model.predict(X_valid)
    print(f"XGBoost Validation Accuracy: {np.mean(y_valid_pred == y_valid)}")
    print(f"XGBoost Validation AUC: {roc_auc_score(y_valid, y_valid_pred)}")

    # Blitz
    ct = BlitzOptimizer(DEPTH=5, D=D, K=2, MAX_ITERS=1, shared_memory=True, verbose=True)
    ct.fit(X_train, y_train)
    y_train_resid = y_train - ct.predict(X_train)

    y_test_pred = ct.predict(X_test)
    print(f"Blitz Test Accuracy: {np.mean(y_test_pred == y_test)}")
    print(f"Blitz Test AUC: {roc_auc_score(y_test,y_test_pred)}")

    y_valid_pred = ct.predict(X_valid)
    print(f"Blitz Validation Accuracy: {np.mean(y_valid_pred == y_valid)}")
    print(f"Blitz Validation AUC: {roc_auc_score(y_valid, y_valid_pred)}")