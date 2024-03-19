ALTER TABLE grants
DROP COLUMN both_names

CREATE TABLE npi (
    npi int,
    taxonomy_code varchar(10),
    last_name varchar(255),
    forename varchar(255),
    address varchar(255),
  	cert_date varchar(255),
  	city varchar(255),
  	state varchar(2),
  	country varchar(100)
);

CREATE TABLE bridge (
  	application_id bigint,
  	npi int,
  	created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);