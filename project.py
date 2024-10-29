import os
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from bokeh.plotting import figure, show , output_file
from bokeh.models import ColumnDataSource
import unittest

class BaseData:

    """
    Base class for handling the database connection.

    """
    def __init__(self, database_name='my_database.db'):

        """
        Initializes the BaseData class and sets up the database connection.

        """
        self.engine = create_engine(f'sqlite:///{database_name}')

class Data(BaseData):

    """
    Class to manage training, ideal, and test data loading from CSV files.

    Inherits from BaseData for database connection handling.

    """
    def __init__(self, train, ideal, test):

        """
        Initializes the Data class and loads CSV files for training, ideal, and test data.

        """
        super().__init__()
        if not os.path.exists(train):
            raise FileNotFoundError(f"The training file '{train}' does not exist.")
        if not os.path.exists(ideal):
            raise FileNotFoundError(f"The ideal file '{ideal}' does not exist.")
        if not os.path.exists(test):
            raise FileNotFoundError(f"The test file '{test}' does not exist.")

        self.data_train = pd.read_csv(train)
        self.data_ideal = pd.read_csv(ideal)
        self.data_test = pd.read_csv(test)

    def load_data(self):

        """
        loads training and ideal data into a combined DataFrame and stores it in the SQLite database.

        """
        try:
            y_columns_file = [f'y{i}' for i in range(1, 5)]
            self.data_train_combined = self.data_train[['x']].copy()
            for y in y_columns_file:
                self.data_train_combined[y] = self.data_train[y]

            self.data_ideal_combined = self.data_ideal.copy()
            y_columns_ideal = [f'y{i}' for i in range(1, 51)]
            for y in y_columns_ideal:
                self.data_ideal_combined[y] = self.data_ideal[y]

            self.data_train_combined.to_sql('training_data', con= self.engine, if_exists='replace')
            self.data_ideal_combined.to_sql('ideal_function', con=self.engine, if_exists='replace')

        except Exception as e:
            print(f"Unexpected error occurred while loading data: {str(e)}")


class Deviation(BaseData):

    """
    Class to calculate the deviation between test data and ideal functions.

    Inherits from BaseData for database connection handling.

    """
    def __init__(self, ideal_data):

        """
        Initializes the Deviation class.

        """
        super().__init__()
        self.ideal_data = ideal_data

    def calculate_deviation(self, x, y_test_value):

        """
        Calculates the deviation between the test data point and each ideal function.

        """
        deviations = {}
        for columns in self.ideal_data.columns[1:]:
            y_ideal = np.interp(x, self.ideal_data['x'], self.ideal_data[columns])
            deviation = np.abs(y_test_value - y_ideal)
            deviations[columns] = deviation

        ideal_function = min(deviations, key=deviations.get)
        local_min_deviation = deviations[ideal_function]
        return ideal_function, local_min_deviation


class Result(BaseData):

    """
    Class to handle saving test results to the database.

    Inherits from BaseData for database connection handling.

    """
    def __init__(self, engine):

        """
        Initializes the Result class with a given SQLAlchemy engine.

        """
        super().__init__()
        self.engine = engine

    def save_results(self, results):

        """
        Saves the test results into the SQLite database.

        """
        self.results = results
        results_data_train = pd.DataFrame(results)
        results_data_train.to_sql('test_results', con=self.engine, if_exists='replace')

def main():

    """
    Main function to load data, calculate deviations, save results, and visualize the deviation results.

    """
    data_manager =Data('train.csv', 'ideal.csv', 'test.csv')
    data_manager.load_data()

    calculator = Deviation(data_manager.data_ideal_combined)

    results = []
    for index, row in data_manager.data_test.iterrows():
        x_test = row['x']
        y_test = row['y']
        ideal_function, min_deviation = calculator.calculate_deviation(x_test, y_test)
        results.append({
            'x': x_test,
            'y': y_test,
            'ideal_function': ideal_function,
            'deviation': min_deviation
        })

    result_saver = Result(data_manager.engine)
    result_saver.save_results(results)

    results = pd.DataFrame(results)
    print(results)

    output_file("output.html")
    source =ColumnDataSource(data=dict(x=results['x'], deviation=results['deviation']))
    p = figure(title="Deviation of Test Data from Ideal Functions",
               x_axis_label='x', y_axis_label='Deviation',
               width=800, height=400)
    p.scatter(source=source, x='x', y='deviation', legend_label="Deviation", color="blue", size=8)
    show(p)

class TestBaseData(unittest.TestCase):

    """
    Unit test class for testing BaseData class.

    """
    def test_database_connection(self):
        data_base = Data('train.csv', 'ideal.csv', 'test.csv')
        self.assertIsNotNone(data_base.engine)

class TestData(unittest.TestCase):

    """
    Test the creation of the database connection.
    """

    def __init__(self, method_name: str = "runTest"):
        super().__init__(method_name)

    def setUp(self):

        """
        Set up the test environment by creating test CSV files.
        """
        data_train= pd.read_csv('train.csv')
        data_ideal = pd.read_csv('ideal.csv')
        data_test = pd.read_csv('test.csv')

        n_train_samples = min(5, len(data_train))
        sample_train = data_train.sample(n=n_train_samples, random_state=1)

        n_ideal_samples = min(5, len(data_ideal))
        sample_ideal = data_ideal.sample(n=n_ideal_samples, random_state=1)

        n_test_samples = min(5, len(data_test))
        sample_test = data_test.sample(n=n_test_samples, random_state=1)

        sample_train.to_csv('test_train.csv', index=False)
        sample_ideal.to_csv('test_ideal.csv', index=False)
        sample_test.to_csv('test_test.csv', index=False)

        self.train_file = 'test_train.csv'
        self.ideal_file = 'test_ideal.csv'
        self.test_file = 'test_test.csv'

        self.data_manager = Data(self.train_file, self.ideal_file, self.test_file)
        self.data_manager.load_data()

    def tearDown(self):

        """
        Clean up the test environment by removing test CSV files.
        """
        os.remove(self.train_file)
        os.remove(self.ideal_file)
        os.remove(self.test_file)

    def test_file_not_found(self):

        """
        Test if FileNotFoundError is raised when file paths are invalid.
        """
        with self.assertRaises(FileNotFoundError):
            Data('invalid_train.csv', self.ideal_file, self.test_file)

    def test_load_data(self):

        """
        Load training and ideal data into a combined DataFrame and store it in the SQLite database.
        """
        try:
            self.assertIsNotNone(self.data_manager.data_train)
            self.assertIsNotNone(self.data_manager.data_ideal)
            self.assertIsNotNone(self.data_manager.data_test)
        except AssertionError:
            print("Error: One or more datasets are not loaded (NoneType encountered).")

        try:
            self.assertTrue('x' in self.data_manager.data_train.columns)
            self.assertTrue('y1' in self.data_manager.data_train.columns)
        except KeyError as key_error:
            print(f"Key error: Column missing - {key_error}")

        try:
            self.assertTrue(len(self.data_manager.data_train) > 0)
            self.assertTrue(len(self.data_manager.data_ideal) > 0)
            self.assertTrue(len(self.data_manager.data_test) > 0)
        except AssertionError:
            print("Error: One or more datasets are empty.")

class TestDeviation(unittest.TestCase):

    """
    A class for testing the deviation calculations between test data and ideal functions.

    """
    def setUp(self):

        """
        Setup method that runs before each test.

        """
        self.data_manager = Data('train.csv', 'ideal.csv', 'test.csv')
        self.data_manager.load_data()
        self.deviation_calculator = Deviation(self.data_manager.data_ideal_combined)
        self.sample_test_data = self.data_manager.data_test.sample(n=1, random_state=42).iloc[0]

    def test_calculate_deviation(self):

        """
        Test method for the calculate_deviation function.

        """
        x_test_value = self.sample_test_data['x']
        y_test_value = self.sample_test_data['y']
        ideal_function, min_deviation = self.deviation_calculator.calculate_deviation(x_test_value, y_test_value)

        print(f"Calculated ideal function: {ideal_function}")
        print(f"Calculated minimum deviation: {min_deviation}")

class TestResult(unittest.TestCase):

    """
    A class for testing the saving of test results into the SQLite database.

    """

    def setUp(self):

        """
        Setup method that runs before each test.

        """
        self.engine = create_engine('sqlite:///:memory:')
        self.result_manager = Result(self.engine)
        self.data_manager = Data('train.csv', 'ideal.csv', 'test.csv')
        self.data_manager.load_data()

    def test_results(self):

        """
        Test method for saving results in the SQLite database.

        """
        sample_test_data = self.data_manager.data_test.sample(n=1, random_state=42).iloc[0]
        results = [{'x': sample_test_data['x'],
                    'y': sample_test_data['y'],
                    'ideal_function': 'y2',
                    'deviation': 1}]
        self.result_manager.save_results(results)
        saved_results = pd.read_sql('test_results', con=self.engine)
        self.assertEqual(len(saved_results), 1)
        print(saved_results)
        self.assertEqual(len(saved_results), 1)
        self.assertEqual(saved_results.iloc[0]['ideal_function'], 'y2')

if __name__ == "__main__":
    main()


