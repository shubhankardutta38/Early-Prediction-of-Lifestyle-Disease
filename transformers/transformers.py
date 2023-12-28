from sklearn.base import BaseEstimator, TransformerMixin

class AgeCategoryTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['AgeCategory'] = X_copy['AgeCategory'].apply(self.process_age_category)
        return X_copy[['AgeCategory']]

    def process_age_category(self, age_category):
        try:
            # Check if the value is a range (e.g., '18-24')
            if '-' in age_category:
                # Extract lower and upper bounds of the range
                lower, upper = map(int, age_category.split('-'))

                # Convert the range to the average value
                return (lower + upper) // 2
            else:
                # Convert a single value to integer
                return int(age_category)
        except ValueError:
            # Handle cases where the conversion fails
            return 80  # Set a default value for cases like '80 or older'

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self
