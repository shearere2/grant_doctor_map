import pandas as pd


class GrantReader():
    def __init__(self, path: str):
        """Pass in path to NIH grants file to clean and read"""
        self.df = self._read(path)

    def to_db(self, con):
        """Send the data to a database via a SQLAlchemy connection"""
        self.df.to_sql('grantee',
                       con,
                       if_exists='append',
                       index=False)

    def _read(self, path: str):
        """Read in a grants NIH file and return a clean dataframe"""
        df = pd.read_csv(path)
        df = df.rename(columns={
            'APPLICATION_ID': 'application_id',
            'BUDGET_START': 'budget_start',
            'ACTIVITY': 'grant_type',
            'TOTAL_COST': 'total_cost',
            'PI_NAMEs': 'pi_names',
            'PI_IDS': 'pi_ids',
            'ORG_NAME': 'organization',
            'ORG_CITY': 'city',
            'ORG_STATE': 'state',
            'ORG_COUNTRY': 'country'
        })

        df = self._clean(df)
        df = df[['application_id',
                 'budget_start',
                 'grant_type',
                 'total_cost',
                 'organization',
                 'city',
                 'state',
                 'country',
                 'forename',
                 'last_name',
                 'is_contact']]
        return df
    
    def _clean(self, df: pd.DataFrame):
        """Clean up the data"""
        # Split apart pi names and make new rows with a single name in each
        df['pi_names'] = df['pi_names'].str.split(';')
        df = df.explode('pi_names')

        # Pull out if the person is the contact
        df['is_contact'] = df['pi_names'].str.lower().str.contains('(contact)', regex=False)
        df['pi_names'] = df['pi_names'].str.replace('(contact)', '')

        # Split apart last and firstnames
        df['both_names'] = df['pi_names'].apply(lambda x: x.split(',')[:2])
        df['last_name'] = df['both_names'].apply(lambda x: x[0])
        df['forename'] = df['both_names'].apply(lambda x: x[1])
        df['last_name'] = df['last_name'].apply(lambda x: x.strip('\''))

        return df
    

if __name__ == '__main__':
    reader = GrantReader('data/RePORTER_PRJ_C_FY2022.csv')
    print(reader.df)