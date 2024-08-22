"""
Instead of creating batches of a single dataset (as Jupyter Notebooks assume), create many different datasets and
    treat each of those as a 'batch'. This file essential runs `sdoh.ipynb` for *.csv/*.jsonl files in a directory.
"""

import concurrent.futures
import csv
import json
import os
from pathlib import Path
import sys
import time

from johnsnowlabs import nlp, medical
from loguru import logger
import sparknlp
import sparknlp_jsl


def run_sdoh(inpath, license_json, hadoop_home=None, note_id_col='note_id', note_text_col='note_text',
             out_dataset=None, logfile='sdoh.log'):
    logger.add(logfile)
    # Load license keys, prepare configurations
    with open(license_json) as f:
        license_keys = json.load(f)

    locals().update(license_keys)
    os.environ.update(license_keys)
    if hadoop_home:
        os.environ['HADOOP_HOME'] = hadoop_home
        os.environ['PATH'] += fr';{hadoop_home}{os.sep}bin;{hadoop_home}'
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    params = {'spark.driver.memory': '4G',
              'spark.executor.memory': '19G',
              'spark.kryoserializer.buffer.max': '2000M',
              'spark.driver.maxResultSize': '2000M',
              }

    spark = sparknlp_jsl.start(license_keys['SECRET'], gpu=False, params=params)

    logger.info(f'Spark NLP Version: {sparknlp.version()}')
    logger.info(f'Spark NLP_JSL Version: {sparknlp_jsl.version()}')

    if not out_dataset:
        out_dataset = Path('sdoh.out.csv')

    jsl_single_pipeline = get_pipeline()

    # write output to CSV
    fieldnames = ['note_id', 'confidence', 'entity', 'assertion']

    note_ner_count = []
    with open(out_dataset, 'w', newline='', encoding='utf8') as fh:
        writer = csv.DictWriter(fh, fieldnames)
        writer.writeheader()
        for in_dataset_path in Path(inpath).iterdir():
            if in_dataset_path.suffix == '.jsonl':
                df = spark.read.json(str(in_dataset_path)).select(note_id_col, note_text_col)
            elif in_dataset_path.suffix == '.csv':
                df = spark.read.csv(str(in_dataset_path)).select(note_id_col, note_text_col)
            else:
                logger.warning(f'Skipping unrecognized extension in file: {in_dataset_path}')
                continue

            # prepare dataset for processing
            df = df.na.drop()  # drop notes without text
            df = df.toDF('note_id', 'note_text')  # rename columns

            # prepare pipelines
            p_model = jsl_single_pipeline.fit(df)
            l_model = nlp.LightPipeline(p_model)

            def annotate(note_id, note_text):
                """ Annotates the data provided into `Annotation` type results. """
                return note_id, l_model.fullAnnotate(note_text)

            before_time = time.time()
            results_list = []

            # run results
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit tasks to the executor
                future_to_data_list = {executor.submit(annotate, note_id, note_text): (note_id, note_text) for
                                       note_id, note_text in df.rdd.map(lambda x: (x.note_id, x.note_text)).collect()}

                # Retrieve results as they complete
                for future in concurrent.futures.as_completed(future_to_data_list):
                    try:
                        results = future.result()  # results: (note_id, result_data)
                        results_list.append(results)
                    except Exception as e:
                        logger.error(f'Generated an exception: {future}')
                        logger.exception(e)
            after_time = time.time()
            logger.info(
                f'{df.count()} notes predicted, {after_time - before_time} seconds, '
                f'avg_predict speed: {(after_time - before_time) / df.count()} s/notes'
            )

            note_ner_count = []
            for note_id, results in results_list:
                for result in results:
                    note_ner_count.append(len(result['filtered']))
                    for entity in result['filtered']:
                        entity = entity.metadata
                        ner_type = entity['entity']
                        assertion = entity['assertion']
                        writer.writerow({
                            'note_id': note_id,
                            'confidence': entity['confidence'],
                            'assertion': assertion,
                            'entity': ner_type,
                        })
            logger.info(
                f'All ner number: {sum(note_ner_count)}, '
                f'avg entity per note: {sum(note_ner_count) / len(note_ner_count)}'
            )


def get_pipeline():
    # the pipeline
    documentAssembler = (
        nlp.DocumentAssembler()
        .setInputCol('note_text')
        .setIdCol('note_id')
        .setOutputCol('document')
    )

    sentenceDetector = (
        nlp.SentenceDetectorDLModel.pretrained('sentence_detector_dl_healthcare', 'en', 'clinical/models')
        .setInputCols(['document'])
        .setOutputCol('sentence').setCustomBounds(['\|'])
    )

    tokenizer = nlp.Tokenizer() \
        .setInputCols(['sentence']) \
        .setOutputCol('token')  # \

    clinical_embeddings = nlp.WordEmbeddingsModel.pretrained('embeddings_clinical', 'en', 'clinical/models') \
        .setInputCols(['sentence', 'token']) \
        .setOutputCol('embeddings')

    ner_model = medical.NerModel.pretrained('ner_sdoh', 'en', 'clinical/models') \
        .setInputCols(['sentence', 'token', 'embeddings']) \
        .setOutputCol('ner')

    ner_conv = medical.NerConverterInternal() \
        .setInputCols(['sentence', 'ner', 'token']) \
        .setOutputCol('chunk_main')

    assertion = medical.AssertionDLModel.pretrained('assertion_sdoh_wip', 'en', 'clinical/models') \
        .setInputCols(['sentence', 'token', 'embeddings', 'chunk_main']) \
        .setOutputCol('assertion')

    assertion_filterer_hypo = medical.AssertionFilterer() \
        .setInputCols(['sentence', 'chunk_main', 'assertion']) \
        .setOutputCol('filtered') \
        .setCriteria('assertion') \
        .setWhiteList(['present', 'Possible', 'Absent', 'Family_History', 'Someone_Else', 'Past'])

    assertion_filterer_present = medical.AssertionFilterer() \
        .setInputCols(['sentence', 'chunk_main', 'assertion']) \
        .setOutputCol('filtered_present') \
        .setCriteria('assertion') \
        .setWhiteList(['present', 'Planned'])

    assertion_filterer_possible = medical.AssertionFilterer() \
        .setInputCols(['sentence', 'chunk_main', 'assertion']) \
        .setOutputCol('filtered_possible') \
        .setCriteria('assertion') \
        .setWhiteList(['Possible'])

    assertion_filterer_absent = medical.AssertionFilterer() \
        .setInputCols(['sentence', 'chunk_main', 'assertion']) \
        .setOutputCol('filtered_absent') \
        .setCriteria('assertion') \
        .setWhiteList(['Absent'])

    assertion_filterer_hist = medical.AssertionFilterer() \
        .setInputCols(['sentence', 'chunk_main', 'assertion']) \
        .setOutputCol('filtered_hist') \
        .setCriteria('assertion') \
        .setWhiteList(['Family_History', 'Someone_Else'])

    assertion_filterer_past = medical.AssertionFilterer() \
        .setInputCols(['sentence', 'chunk_main', 'assertion']) \
        .setOutputCol('filtered_past') \
        .setCriteria('assertion') \
        .setWhiteList(['Past'])

    jsl_single_pipeline = nlp.Pipeline(
        stages=[
            documentAssembler,
            sentenceDetector,
            tokenizer,
            clinical_embeddings,
            ner_model,
            ner_conv,
            assertion,
            assertion_filterer_hypo,
            assertion_filterer_present,
            assertion_filterer_possible,
            assertion_filterer_absent,
            assertion_filterer_hist,
            assertion_filterer_past
        ]
    )
    return jsl_single_pipeline


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--license-json', dest='license_json', required=True,
                        help='Path to `spark_jsl_VV.json` download from `https://my.johnsnowlabs.com/subscriptions`.')
    parser.add_argument('--hadoop-home', dest='hadoop_home', default=None,
                        help='Hadoop home and add to path.')
    parser.add_argument('--note-id-col', dest='note_id_col', default='note_id',
                        help='Variable name for note_id (unique identifier for note).')
    parser.add_argument('--note-text-col', dest='note_text_col', default='note_text',
                        help='Variable name for note\'s text.')
    parser.add_argument('--inpath', dest='inpath', required=True,
                        help='Path to directory containing dataset files (each is a batch). Must be CSV or JSONL.')
    parser.add_argument('--out-dataset', dest='out_dataset', default=None,
                        help='Path to output CSV file.')
    parser.add_argument('--logfile', default='sdoh.log')
    args = parser.parse_args()
    run_sdoh(**vars(args))
