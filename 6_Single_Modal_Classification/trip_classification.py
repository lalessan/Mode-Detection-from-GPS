import sys
sys.path.insert(0, "/home/s232534/scripts")

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from multiprocessing import cpu_count

def encode_and_split(df: pd.DataFrame) -> tuple[list, list, list, list, list]:
    classifier_col = "MAIN_MODE"        
    drop_cols = ['TRIP_KEY', 'MAIN_MODE']

    # X and y 
    y = df[classifier_col].astype("category")
    X = df.drop(columns=drop_cols)

    # Since our labels are text we need to encode them as numerical catergories
    le = LabelEncoder()
    y_e = le.fit_transform(y)
    class_names = le.classes_.tolist()

    # Split the data and only allow the opt to use train data so that we protect the test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_e,
        test_size=0.2,
        stratify=y_e,
        random_state=0
    )
    
    return X_train, X_test, y_train, y_test, class_names

def run_xgb_classifier(df) -> tuple[XGBClassifier, list, list, list]:
    X_train, X_test, y_train, y_test, class_names = encode_and_split(df)
    
    number_of_classes = len(class_names)
    xgb = XGBClassifier(
        objective='multi:softprob',
        num_class=number_of_classes,
        tree_method='hist',
        n_jobs=cpu_count(),
        random_state=0,
        eval_metric='mlogloss')

    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    
    return xgb, y_pred, y_test, class_names

def run_xgb_classifier_kfold(df, splits: int = 5, return_le=False):
    classifier_col = "MAIN_MODE"        
    drop_cols = ['TRIP_KEY', 'MAIN_MODE']

    # X and y 
    X = df.drop(columns=drop_cols)
    y = df[classifier_col].astype("category")

    number_of_classes = y.cat.categories.size

    kf = KFold(n_splits=splits, shuffle=True, random_state=0)

    le = LabelEncoder()
    y_e = le.fit_transform(y)
    class_names = le.classes_.tolist()
    num_classes = len(class_names)    
    
    f1_model = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_e[train_idx], y_e[test_idx]
        
        xgb_model = XGBClassifier(
            objective='multi:softprob',
            num_class=number_of_classes,
            tree_method='hist',
            n_jobs=cpu_count(),
            random_state=0,
            eval_metric='mlogloss')

        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        
        f1_fold = f1_score(y_test, y_pred, average="macro")
        f1_model.append((f1_fold, xgb_model))

    best_f1, best_model = max(f1_model, key=lambda t: t[0])
    
    if return_le:
        return best_f1, best_model, le
    else:
        return best_f1, best_model

# --------------------------------------------------------------------
# -- The following are used for visualisation of classifier results --
# --------------------------------------------------------------------

def recall(cm, row_sums):
    # TP / (TP + FN)
    TP = np.diag(cm)
    FN = row_sums - TP
    
    recall_val = TP / (TP + FN)
    
    return recall_val * 100

def precision(cm, col_sums):
    # TP / (TP + FN)
    TP = np.diag(cm)
    FP = col_sums - TP
    
    recall_val = TP / (TP + FP)
    
    return recall_val * 100

def f1_score_gen(precision, recall):
    # 2 * (Precision*Recall / Precision+Recall)
    return 2 * ((precision * recall) / (precision + recall))

def accuracy(cm, col_sums):
    # num correct predictions / total number of samples
    # Can also just use sklearns accuracy_score...
    total = col_sums.sum()
    correct = np.diag(cm).sum()
    
    return (correct / total) * 100   

def generate_confusion_table(cm, class_names) -> pd.DataFrame:
    # Build the df
    cm_df = pd.DataFrame(
        cm,
        index=pd.Index(class_names, name="Actual class"),
        columns=pd.Index(class_names, name="Predicted class")
    )
    
    row_sums = cm_df.sum(axis=1).values
    col_sums = cm_df.sum(axis=0).values

    cm_df["Row Sum"] = row_sums
    cm_df.loc["Col Sum"] = cm_df.sum(axis=0).values

    recall_arr = recall(cm, row_sums)
    precision_arr = precision(cm, col_sums)

    # Recall and Precision
    cm_df["Row Recall"] = np.append(recall_arr, np.nan)
    cm_df.loc["Col Precision"] = np.append(precision_arr, [np.nan, np.nan])

    # F1 Score
    cm_df.loc["F1 Score"] = f1_score_gen(cm_df.loc["Col Precision"], cm_df["Row Recall"])

    # Accuracy
    cm_df.loc["Accuracy"] = np.nan
    cm_df.loc["Accuracy", "Row Sum"] = accuracy(cm, col_sums)

    # Append sum of F1 mean!
    cm_df.loc["F1 Score", "Row Sum"] = cm_df.loc["F1 Score"].sum() / cm.shape[0]

    return cm_df

def generate_combined_difference_table(cm1: np.array, cm2: np.array, df: pd.DataFrame, class_names: list, df2: pd.DataFrame = None) -> pd.DataFrame:
    """Generate a dataframe that can be converted to a table that contains the target confusion matrix values and the difference 
    of values from the previous confusion matrix. E.g., cm1 val = 20, cm2 val = 15, diff = 5 so then one cell in our confusion section
    of the table will be cm1 = 20 (+5).

    Args:
        cm1 (np.array): _description_
        cm2 (np.array): _description_
        df (pd.DataFrame): _description_
        class_names (list): _description_

    Returns:
        pd.DataFrame: _description_
    """
    diff_cm = cm1 - cm2
    combined_cm = cm1.astype(str) + " (" + diff_cm.astype(int).astype(str) + ")"

    if df2 is not None:
        diff_df = df - df2
        
        df_out = df.copy().astype("object")
        
        df_out.loc[class_names, "Row Sum"] = (
            df.loc[class_names, "Row Sum"].astype(int).astype(str)
            + " ("
            + diff_df.loc[class_names, "Row Sum"].astype(int).astype(str)
            + ")"
        )
        df_out.loc[class_names, "Row Recall"] = (
            df.loc[class_names, "Row Recall"].round(3).astype(str)
            + " ("
            + diff_df.loc[class_names, "Row Recall"].round(3).astype(str)
            + ")"
        )
        
        for r, decimals in [("Col Sum", 0), ("Col Precision", 3), ("F1 Score", 3), ("Accuracy", 3)]:
            if r in df.index:
                left = df.loc[r, class_names].round(decimals)
                right = diff_df.loc[r, class_names].round(decimals)
                if decimals == 0:
                    if left is not np.nan:
                        df_out.loc[r, class_names] = left.astype(int).astype(str) + " (" + right.astype(int).astype(str) + ")"
                else:
                    if left is not np.nan:
                        df_out.loc[r, class_names] = left.astype(str) + " (" + right.astype(str) + ")"
        
    if df2 is not None:
        cm_combined_df = df_out
    else:
        cm_combined_df = df.copy()
        
    cm_combined_df = cm_combined_df.astype("object") # Since we are setting the inner matrix that is float to str we move to obj datatype to avoid warnings/errors

    # Overwrite the Class matrix only:
    cm_combined_df.loc[class_names, class_names] = combined_cm
    
    return cm_combined_df

def generate_combined_difference_table_all(cm1: np.array, cm2: np.array, df: pd.DataFrame, class_names: list, df2: pd.DataFrame = None) -> pd.DataFrame:
    diff_cm = cm1 - cm2
    combined_cm = cm1.astype(str) + " (" + diff_cm.astype(int).astype(str) + ")"

    # base output
    df_out = df.copy().astype("object")

    # if no comparison df, just return with confusion block overwritten
    if df2 is None:
        df_out.loc[class_names, class_names] = combined_cm
        return df_out

    diff_df = df - df2

    def combine_series(left: pd.Series, right: pd.Series, decimals: int, as_int: bool) -> pd.Series:
        out = pd.Series(np.nan, index=left.index, dtype="object")
        mask = left.notna() & right.notna()

        if as_int:
            out.loc[mask] = (
                left.loc[mask].round(0).astype(int).astype(str)
                + " ("
                + right.loc[mask].round(0).astype(int).astype(str)
                + ")"
            )
        else:
            out.loc[mask] = (
                left.loc[mask].round(decimals).astype(str)
                + " ("
                + right.loc[mask].round(decimals).astype(str)
                + ")"
            )
        return out

    # update summary columns for actual-class rows
    if "Row Sum" in df.columns:
        df_out.loc[class_names, "Row Sum"] = combine_series(
            df.loc[class_names, "Row Sum"], diff_df.loc[class_names, "Row Sum"],
            decimals=0, as_int=True
        )

    if "Row Recall" in df.columns:
        df_out.loc[class_names, "Row Recall"] = combine_series(
            df.loc[class_names, "Row Recall"], diff_df.loc[class_names, "Row Recall"],
            decimals=3, as_int=False
        )

    # update summary rows across predicted-class columns
    for r, decimals, as_int in [("Col Sum", 0, True), ("Col Precision", 3, False), ("F1 Score", 3, False), ("Accuracy", 3, False)]:
        if r in df.index:
            df_out.loc[r, class_names] = combine_series(
                df.loc[r, class_names], diff_df.loc[r, class_names],
                decimals=decimals, as_int=as_int
            )

            # also update the metric's "total" cell in Row Sum (this is where Accuracy/F1 total usually are)
            if "Row Sum" in df.columns:
                left = df.loc[r, "Row Sum"]
                right = diff_df.loc[r, "Row Sum"]
                if pd.notna(left) and pd.notna(right):
                    if as_int:
                        df_out.loc[r, "Row Sum"] = f"{int(round(left))} ({int(round(right))})"
                    else:
                        df_out.loc[r, "Row Sum"] = f"{round(left, decimals)} ({round(right, decimals)})"

    # overwrite the confusion-matrix block
    df_out.loc[class_names, class_names] = combined_cm
    return df_out

def train_and_evaluate(feature, XGBmodel, X_train, X_test, y_train, y_test):
    X_train_pair = X_train[feature]
    XGBmodel.fit(X_train_pair, y_train)
    
    X_test_pair = X_test[feature]
    
    y_pred = XGBmodel.predict(X_test_pair)
    accuracy = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro')

    inner_results = {
        'accuracy': accuracy,
        'f1_macro': f1,
        #'classification_report': classification_report(y_test, y_pred, labels=np.arange(len(class_names)), target_names=class_names)
    }
    
    return inner_results

def forward_feature_selection_simple(
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    y_test,
    xgb_params: dict,
) -> list[str]:
    all_feature_names: list[str] = list(X_train.columns)
    selected_feature_names: list[str] = []
    remaining_feature_names: list[str] = all_feature_names.copy()

    best_overall_score: float | None = None

    for step_index in range(len(all_feature_names)):
        print(f"\nForward selection step {step_index + 1}...")
        best_feature_this_round: str | None = None
        best_score_this_round: float | None = None

        # Try adding each remaining feature
        for candidate_feature_name in remaining_feature_names:
            # Stack our features here so it always include the feature from the prev round
            candidate_feature_set = selected_feature_names + [candidate_feature_name]

            X_train_candidate = X_train[candidate_feature_set]
            X_test_candidate = X_test[candidate_feature_set]

            xgb_model = XGBClassifier(**xgb_params)
            xgb_model.fit(X_train_candidate, y_train)

            y_test_predicted = xgb_model.predict(X_test_candidate)

            # F1 macro scoring
            candidate_score = f1_score(
                y_test,
                y_test_predicted,
                average="macro",
            )

            print(
                f"  Tried adding '{candidate_feature_name}': "
                f"macro F1 = {candidate_score:.4f}"
            )

            if (best_score_this_round is None) or (candidate_score > best_score_this_round):
                best_score_this_round = candidate_score
                best_feature_this_round = candidate_feature_name

        # Nothing improved (shouldn't really happen, but just in case)
        if best_feature_this_round is None:
            print("No candidate feature found. Stopping.")
            break

        # From 1st round onwards if no improvment (better F1 score) is found then we stop.
        if (best_overall_score is not None) and (best_score_this_round <= best_overall_score):
            print(
                f"No improvement over previous best score "
                f"({best_overall_score:.4f}). Stopping."
            )
            break

        # Accept this feature
        selected_feature_names.append(best_feature_this_round)
        remaining_feature_names = [
            f for f in remaining_feature_names if f != best_feature_this_round
        ]
        best_overall_score = best_score_this_round

        print(
            f"Selected feature '{best_feature_this_round}' | "
            f"current best macro F1 = {best_overall_score:.4f}"
        )
        print(f"Current selected feature set: {selected_feature_names}")

        if not remaining_feature_names:
            print("No remaining features left. Stopping.")
            break

    return selected_feature_names

# DEPRACTED / UNUSED:
def plot_confusion_matrix(y_test, y_pred, class_names, title: str = "Confusion Matrix for..."):
    cm = metrics.confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))

    #im = ax.imshow(cm, cmap="Blues")

    # Add grid lines between cells
    ax.set_xticks(np.arange(cm.shape[1]+1)-0.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Tick labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # draw a light-green rectangle on the diagonal
            if i == j:
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color="#b6f5b6", alpha=0.6)
                ax.add_patch(rect)
            ax.text(j, i, cm[i, j],
                    ha="center", va="center", color="black")
    
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    
    plt.title(title, pad=20)
    plt.tight_layout()
    plt.show()