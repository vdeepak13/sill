use paraml;

CREATE TABLE cpuusage (
  graph VARCHAR(100), KEY(graph),
  ncpus INT, KEY(ncpus),
  pfactor INT, KEY(pfactor),
  cpuid INT,
  clock DOUBLE,
  event INT
);

CREATE TABLE networkusage (
  graph VARCHAR(100), KEY(graph),
  ncpus INT, KEY(ncpus),
  pfactor INT, KEY(pfactor),
  cpuid INT,
  send INT,
  recv INT
);

CREATE TABLE summary (
  graph VARCHAR(100), KEY(graph),
  ncpus INT, KEY(ncpus),
  pfactor INT, KEY(ncpus),
  runtime DOUBLE,
  totaltime DOUBLE,
  energy DOUBLE,
  likelihood DOUBLE
);

