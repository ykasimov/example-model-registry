from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from mlem.api import save


def main():
    data, y = load_iris(return_X_y=True, as_frame=True)
    rf = RandomForestClassifier(
        n_estimators=10,
        n_jobs=2,
        random_state=42,
    )
    rf.fit(data, y)

    save(
        rf,
        "models/my_rf",
        sample_data=data,
        external=True,
        description="my rf model",
    )


if __name__ == "__main__":
    main()