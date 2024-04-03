import sqlalchemy

from grant_doctor_map.readers import grants, npi


engine = sqlalchemy.create_engine('sqlite:///data/grant_npi.db')
conn = engine.connect()

greader = grants.GrantReader('data/RePORTER_PRJ_C_FY2022.csv')
greader.to_db(conn)

nreader = npi.NPIReader('data/npidata_pfile_20240205-20240211.csv')
nreader.to_db(conn)