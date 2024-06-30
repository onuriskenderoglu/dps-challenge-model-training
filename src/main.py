import os.path
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def main():
    df = preprocess_data()

    x, y = df.drop('WERT', axis=1), df['WERT']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    forest = RandomForestRegressor()
    forest.fit(x_train, y_train)
    forest.score(x_test, y_test)

    with open('final_model.pkl', 'wb') as f:
        pickle.dump(forest, f)


def preprocess_data():
    data_path = os.path.join("..", "data", "monatszahlen2405_verkehrsunfaelle_export_31_05_24_r.csv")
    df = pd.read_csv(data_path)
    df = df[df['JAHR'] < 2021]
    df = df[['MONATSZAHL', 'AUSPRAEGUNG', 'JAHR', 'MONAT', 'WERT']]

    # One-hot encoding for categorical string data fields
    df = df.join(pd.get_dummies(df['MONATSZAHL'], dtype=int)).drop(['MONATSZAHL'], axis=1)
    df = df.join(pd.get_dummies(df['AUSPRAEGUNG'], dtype=int)).drop(['AUSPRAEGUNG'], axis=1)

    df.dropna(inplace=True)
    df = df[pd.to_numeric(df['MONAT'], errors='coerce').notnull()]
    return df


if __name__ == "__main__":
    main()