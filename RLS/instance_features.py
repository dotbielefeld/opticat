import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib



class Feature_Transfomer:
    def __init__(self, delete_features=True):
        self.threshold = 10  # Threshold for the number of unique values to be considerd cat feature

        self.onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.standard_scaler = StandardScaler()
        self.delete_features = delete_features
        self.transformer = None  # This will hold the fitted transformer
        self.variable_value_columns = None

    def fit_transform(self, features):

        if self.delete_features:
            # drop colums where features are the same
            self.variable_value_columns = np.any(features != features[0, :], axis=0)
            features_all_diff = features[:, self.variable_value_columns]

            categorical_columns = [i for i in range(features_all_diff.shape[1])
                                   if len(np.unique(features_all_diff[:, i])) <= self.threshold]

            numerical_columns = [i for i in range(features_all_diff.shape[1])
                                 if i not in categorical_columns]

            # Column Transformer to apply the transformations
            self.transformer = ColumnTransformer(
                [("cat", self.onehot_encoder, categorical_columns),
                 ("num", self.standard_scaler, numerical_columns)],
                remainder='passthrough')
            features_transformed = self.transformer.fit_transform(features_all_diff)
        else:
            categorical_columns = [i for i in range(features.shape[1])
                                   if len(np.unique(features[:, i])) <= self.threshold]

            numerical_columns = [i for i in range(features.shape[1])
                                 if i not in categorical_columns]

            # Column Transformer to apply the transformations
            self.transformer = ColumnTransformer(
                [("cat", self.onehot_encoder, categorical_columns),
                 ("num", self.standard_scaler, numerical_columns)],
                remainder='passthrough')

            features_transformed = self.transformer.fit_transform(features)

        return features_transformed

    def transform(self, features):
        if self.delete_features and self.variable_value_columns is not None:
            features = features[:, self.variable_value_columns]
        return self.transformer.transform(features)

    def save_transformer(self, filepath):
        # Save the state of the transformer
        import joblib
        joblib.dump({
            "transformer": self.transformer,
            "variable_value_columns": self.variable_value_columns
        }, filepath)

    def load(self, filepath):
        # Load the state of the transformer
        import joblib
        state = joblib.load(filepath)
        self.transformer = state["transformer"]
        self.variable_value_columns = state["variable_value_columns"]
