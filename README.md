# FE5 JSL

Running JohnSnowLabs for the relevant FE5 variables: SDOH.

These files will require a JohnSnowLabs license which is not free. I'm not quite sure which license is required but `nlp` and `medical` from `johnsnowlabs` are imported.

Many of these have been included as both Jupyter Notebooks and raw Python files. These have been particularly setup for a Windows environment -- running on Linux should be easier and the relevant files can be cannabilized.

## Prerequisites

* Install a version of Python compatible with Apache Spark/PySpark
  * `loguru`: `pip install loguru` (for improved logging)
* Follow the steps to install Python prerequisites: https://my.johnsnowlabs.com/docs > `Install Locally on Python`
  * Running `sparknlp_jsl.start` will take a lot of time the first time running due to need to install dependent jars
* Download license keys as json from `https://my.johnsnowlabs.com/subscriptions`
  * Save as `spark_jsl_VV.json` where VV is the version number (e.g., 5.4). Each version requires a different config file
* Spark / Hadoop (?)
  * See `Windows Prereqs` for what's required on Windows
* JDK 8/1.8
  * Add to `PATH` and set `JAVA_HOME` to install path

### On Windows

* Download relevant version of hadoop/bin to, e.g., `C:\hadoop\bin` from https://github.com/cdarlint/winutils 
* Environment variables (these may be included in the code blocks):
    * `HADOOP_HOME`= `C:\hadoop`
    * `PATH` += `%HADOOP_HOME%\bin`

## SDOH

* See `nbs/sdoh.ipynb` for running a single file in memory.
* See `nbs/sdoh_iter.ipynb` for running a single file in batches (specify batch_size in config options.
  * This is recommended.
* See `src/run_sdoh_batches.py` for running against a directory of datasets, where each dataset can fit in memeory
  * Not tested...not sure it's the best idea.

