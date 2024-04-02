import pandas as pd
import db,model_features


def sample_last_names():
    """Get a sample of last names from both databases
    """
    df = pd.read_sql('''SELECT DISTINCT gr.last_name
                        FROM grants gr
                        INNER JOIN npi
                            ON gr.last_name = npi.last_name
                        LIMIT 100;''', db.sql())
    df = df.loc[~df['last_name'].str.contains("'")]
    return df


def get_probable_matches():
    """Get a set of likely matches between grantee/grant and 
    provider/npi. We will use distances to estimate likely matches."""
    sample = sample_last_names()
    sample['last_name'] = "'" + sample['last_name'] + "'"
    names = ', '.join(sample['last_name'])
    query = f'''SELECT forename,last_name,organization,city,state,country
            FROM grants
            WHERE last_name IN ({names});'''
    grantees = pd.read_sql(query, db.sql()).add_prefix('g_').rename(columns={'g_last_name':'last_name'})
    query = f'''SELECT forename,last_name,city,state,country
            FROM npi
            WHERE last_name IN ({names});'''
    npi = pd.read_sql(query,db.sql()).add_prefix('n_').rename(columns={'n_last_name':'last_name'})
    
    comb = grantees.merge(npi,on='last_name')
    comb['forename_jw_dist'] = comb.apply(
        lambda row: model_features.jw_dist(
            row['g_forename'],row['n_forename']), axis=1)
    return comb.sort_values(
        by='forename_jw_dist',ascending=False).groupby(
            'last_name')['forename_jw_dist'].head(5)


if __name__ == '__main__':
    df = get_probable_matches()
    df.to_csv('data/likely_grantee_provider_matches.csv',index=False)