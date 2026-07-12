import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def display_probabilities(model, new_data, title):
    probabilities = model.predict_proba(new_data)[0]

    print(title)
    print("-" * len(title))

    for class_name, probability in zip(model.classes_, probabilities):
        print(f"{class_name}: {probability * 100:.2f}%")


def main():
    data_file = Path("data") / "student_results.csv"
    output_folder = Path("outputs")
    output_file = output_folder / "tree_forest_results.csv"
    tree_chart_file = output_folder / "decision_tree.png"
    importance_file = output_folder / "feature_importance.csv"

    output_folder.mkdir(exist_ok=True)

    df = pd.read_csv(data_file)

    print("Student Result Dataset")
    print("----------------------")
    print(df)

    print()
    print("Result Counts")
    print("-------------")
    print(df["Result"].value_counts())

    X = df[["StudyHours", "Attendance", "AssignmentScore"]]
    y = df["Result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    tree_model = DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    )

    forest_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    tree_model.fit(X_train, y_train)
    forest_model.fit(X_train, y_train)

    tree_predictions = tree_model.predict(X_test)
    forest_predictions = forest_model.predict(X_test)

    tree_accuracy = accuracy_score(y_test, tree_predictions)
    forest_accuracy = accuracy_score(y_test, forest_predictions)

    results = pd.DataFrame({
        "StudyHours": X_test["StudyHours"],
        "Attendance": X_test["Attendance"],
        "AssignmentScore": X_test["AssignmentScore"],
        "ActualResult": y_test,
        "DecisionTreePrediction": tree_predictions,
        "RandomForestPrediction": forest_predictions
    })

    print()
    print("Prediction Results")
    print("------------------")
    print(results)

    print()
    print("Model Accuracy")
    print("--------------")
    print(f"Decision Tree Accuracy: {tree_accuracy:.2f}")
    print(f"Random Forest Accuracy: {forest_accuracy:.2f}")

    new_student = pd.DataFrame({
        "StudyHours": [6],
        "Attendance": [78],
        "AssignmentScore": [70]
    })

    tree_new_prediction = tree_model.predict(new_student)
    forest_new_prediction = forest_model.predict(new_student)

    print()
    print("New Student")
    print("-----------")
    print(new_student)

    print()
    print(f"Decision Tree Prediction: {tree_new_prediction[0]}")
    print(f"Random Forest Prediction: {forest_new_prediction[0]}")

    print()
    display_probabilities(
        tree_model,
        new_student,
        "Decision Tree Probabilities"
    )

    print()
    display_probabilities(
        forest_model,
        new_student,
        "Random Forest Probabilities"
    )

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": forest_model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    print()
    print("Feature Importance")
    print("------------------")
    print(feature_importance)

    results.to_csv(output_file, index=False)
    feature_importance.to_csv(importance_file, index=False)

    plt.figure(figsize=(12, 8))

    plot_tree(
        tree_model,
        feature_names=X.columns,
        class_names=tree_model.classes_,
        filled=True
    )

    plt.title("Decision Tree")
    plt.tight_layout()
    plt.savefig(tree_chart_file)
    plt.show()

    print()
    print(f"Prediction results saved to: {output_file}")
    print(f"Feature importance saved to: {importance_file}")
    print(f"Decision tree chart saved to: {tree_chart_file}")


main()
