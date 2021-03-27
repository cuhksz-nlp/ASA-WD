# ASA-WD

This is the implementation of [Enhancing Aspect-level Sentiment Analysis with Word Dependencies]() at EACL 2021.

You can e-mail Guimin Chen at `chenguimin@sribd.cn` or `cuhksz.nlp@gmail.com`, if you have any questions.

## Citation

If you use or extend our work, please cite our paper at COLING 2020.

```
@inproceedings{
}
```

## Requirements

Our code works with the following environment.
* `python=3.7`
* `pytorch=1.3`

## Dataset

To obtain the data, you can go to [`data`](./data) directory for details.

## Downloading BERT and ASA-WD

In our paper, we use BERT ([paper](https://www.aclweb.org/anthology/N19-1423/)) as the encoder.

For BERT, please download pre-trained BERT-Base and BERT-Large English from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

For ASA-WD, you can download the models we trained in our experiments from [Google Drive]() or [Baidu Net Disk]() .

## Run on Sample Data

Run `run_sample.sh` to train a model on the small sample data under the `sample_data` directory.

## Training and Testing

You can find the command lines to train and test models in `run_train.sh` and `run_test.sh`, respectively.

Here are some important parameters:

* `--do_train`: train the model.
* `--do_eval`: test the model.

## To-do List

* Release the code to get the data.
* Regular maintenance.

You can leave comments in the `Issues` section, if you want us to implement any functions.

