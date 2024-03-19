import pandas as pd


def read(path: str) -> pd.DataFrame:
    """Read in an npi file from path, clean, and return"""
    df = pd.read_csv(path)
    mapper = {
            'NPI': 'npi',
            'Healthcare Provider Taxonomy Code_1': 'npi_taxonomy_code',
            'Provider Last Name (Legal Name)': 'npi_last_name',
            'Provider First Name': 'npi_forename',
            'Provider First Line Business Practice Location Address': 'npi_address',
            'Certification Date': 'npi_cert_date',
            'Provider Business Practice Location Address State Name': 'npi_city',
            'Provider Business Practice Location Address State Name': 'npi_state',
            'Provider Business Practice Location Address Country Code (If outside U.S.)': 'npi_country'
        }
    
    df = df.rename(columns=mapper)[mapper.values()]
    return df


if __name__ == '__main__':
    df = read("data/npidata_pfile_20240205-20240211.csv")
    print(df.head())