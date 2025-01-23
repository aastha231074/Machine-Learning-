import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """
    A class to handle all data preprocessing steps including:
    - Loading data
    - Handling missing values
    - Encoding categorical variables
    - Splitting dataset
    - Feature scaling
    """
    
    def __init__(self, test_size=0.2, random_state=0):
        """
        Initialize the DataPreprocessor with splitting parameters.
        
        Parameters:
        -----------
        test_size : float, default=0.2
            Proportion of the dataset to include in the test split
        random_state : int, default=0
            Controls the shuffling applied to the data before applying the split
        """
        self.test_size = test_size
        self.random_state = random_state
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """
        Load data from CSV file and separate features and target.
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
            
        Returns:
        --------
        tuple : (features array, target array)
        """
        data = pd.read_csv(filepath)
        self.X = data.iloc[:, :-1].values
        self.y = data.iloc[:, -1].values
        return self.X, self.y
    
    def handle_missing_values(self):
        """
        Handle missing values in the dataset using mean strategy.
        """
        self.X[:, 1:-1] = self.imputer.fit_transform(self.X[:, 1:-1])
        
    def encode_categorical_variables(self):
        """
        Encode categorical variables using Label Encoding and One Hot Encoding.
        """
        # Label encoding for first column of X
        self.X[:, 0] = self.label_encoder.fit_transform(self.X[:, 0])
        
        # One hot encoding for first column
        self.X = self.one_hot_encoder.fit_transform(self.X[:, [0]])
        
        # Label encoding for target variable
        self.y = self.label_encoder.fit_transform(self.y)
        
    def split_data(self):
        """
        Split the dataset into training and test sets.
        
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test)
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, 
            self.y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """
        Scale features using StandardScaler.
        
        Returns:
        --------
        tuple : (scaled X_train, scaled X_test)
        """
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        return self.X_train, self.X_test
    
    def preprocess(self, filepath):
        """
        Execute all preprocessing steps in sequence.
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
            
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test)
        """
        self.load_data(filepath)
        self.handle_missing_values()
        self.encode_categorical_variables()
        self.split_data()
        self.scale_features()
        return self.X_train, self.X_test, self.y_train, self.y_test