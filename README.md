# Social Computing
This is the repository of Social Computing Final Assignment submission titled *Gender Confrontation on the Chinese Internet: Analysis Based on Baidu Index and Weibo Comments*.



## Setup

This implementation is based on Python 3. To run the code, you need the following dependencies.

- jieba==0.42.1

- matplotlib==3.8.4

- numpy==1.24.3

- pandas==1.5.3

- scikit_learn==1.4.2

- torch==2.3.0

- transformers==4.42.2

- wordcloud==1.9.3

To install the dependencies, you may use pip as follows.

```python
pip install -r requirements.txt
```

If you still cannot run the code, please try changing the version of these packages.



## Repository structure

We select some important files for detailed description.

```python
|-- blog-Id                             # for getting blog ids to crawl the comments under them
|-- CORGI-PM                            # for bias detection
    |-- dataset
    	|-- bias_sentence.npy           # biased sentences from Weibo
        |-- CORGI-PC_splitted_biased_corpus_v1.npy
        |-- CORGI-PC_splitted_non-bias_corpus_v1.npy
        |-- unbias_sentence.npy         # unbiased sentences from Weibo
    |-- src
    	|-- run_classification.py       # bias detection code
|-- emotion.ipynb                       # sentiment analysis
|-- k_means.ipynb                       # embedding clustering
|-- test_making.ipynb                   # for generating real dataset for bias detection
|-- wordcloud.ipynb                     # word cloud
|-- bert_dnn_8.model                    # BERT based on Weibo data
|-- comment_20240626225604.jsonl        # crawled Weibo comments
|-- filtered_tweets_sorted_copy.jsonl   # crawled Weibo posts
|-- numpy_list.npy                      # embedding vectors of Weibo comments
|-- requirements.txt                    # dependencies
|-- LazyCSS社会计算期末项目报告.pdf      # report
|-- Oral Defence.pptx                   # slides for oral defence
```



## Run our code

For all the .ipynb files, you can simply run them in Jupyter Notebooks.

If you want to run the code for bias detection, you can run our code as in the script in the below:

```python
python -u ./CORGI-PM/src/run_classification.py
```
