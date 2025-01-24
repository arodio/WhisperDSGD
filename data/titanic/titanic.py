import torch
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from torch.utils.data import Dataset


class TitanicDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # Load data
        data = pd.read_csv(csv_file)

        # Separate target and features
        features = data.drop('Survived', axis=1)
        targets = data['Survived']

        # Define preprocessing pipeline
        numeric_features = ['Age', 'Fare', 'Pclass']
        categorical_features = ['Sex', 'Embarked']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Preprocess data
        data = preprocessor.fit_transform(features)
        targets = targets.values

        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            features = self.transform(features)
        return features, label
