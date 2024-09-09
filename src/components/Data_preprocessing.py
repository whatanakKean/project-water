import pandas as pd

class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def preprocess_data(self):
        df_filtered = self.df[['date_gmt', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']]
        df_filtered.set_index('date_gmt', inplace=True)
        df_filtered.reset_index(inplace=True)
        df_long = pd.melt(df_filtered, id_vars=['date_gmt'], value_vars=['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024'],
                        var_name='Year', value_name='water_level')
        df_long['DATE_GMT'] = pd.to_datetime(df_long['date_gmt'] + '-' + df_long['Year'], format='%m-%d-%Y').dt.strftime('%m-%d-%Y')
        df_long = df_long[['DATE_GMT', 'water_level']]
        df_non_zero = df_long[df_long['water_level'] != 0]
        df_non_zero.set_index('DATE_GMT', inplace=True)
        df_non_zero.index.freq='D'

        df_non_zero.to_csv('../data/data_preprocessed_PPB.csv')
        return df_non_zero