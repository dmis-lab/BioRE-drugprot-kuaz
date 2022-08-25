## Relation Extraction model (Our participation at the BioCreative 7 - Drugprot challenge)

In this repository, we provide source codes and resources of our participation at the BioCreative 7 DrugProt challenge.

<hr>

### Requirements

Codes are tested using Python v3.7.2 and following libraries.

```
torch>=1.3.1
transformers==4.9.0
datasets==1.8.0
```

Our main code `run_re_hfv4.py` is based on [example codes](https://github.com/huggingface/transformers/blob/v4.9.2/examples/pytorch/text-classification/run_glue.py) of transformer repository, with modification for our pre-processing style and use case.

## Pre-processed datasets

Please download pre-processed datasets from [here](https://drive.google.com/file/d/1XQ-hjYO01XUqyBl4C9fWIg_IzZ8YVtAg/view?usp=sharing)

The compressed file contains pre-processed `train`, `development` and `test` datasets.

#### NOTE: To evaluate the model predictions on the development set: 
Please replace `test-mapping.tsv`(6.9M) and `test.tsv`(67M) with the files from `dev_named_as_test` folder (`dev_named_as_test/test-mapping.tsv`(399K), `dev_named_as_test/test-mapping.tsv`(3.9M)). 
Those two files are the developement dataset pre-processed and renamed in the format of test dataset, to make predictions without modifying `run_re_hfv4.py`.

The pre-processed test.tsv should have 238,624 lines whereas the developement data dev.tsv (which is identical to `dev_named_as_test/test.tsv` in the content) should have 13,480 lines. (Measured using `wc -l`)
The corresponding `test-mapping.tsv` files should have the same number of lines.


## How to train the model / make predictions using trained model


First, train your model using `run_re_hfv4.py`.

For example, the following code (linux bash script) will produce prediction results and checkpoints in `$OUTPUT_DIR`.
```bash
export SEED=0
export CASE_NUM=`printf %02d $SEED`

export LM_FULL_NAME=<LM PATH or HF Transformer name/url>
export SEQ_LEN=192
export BATCH_SIZE=16 #16 with LR 2e-5  #32 with LR 5e-5
export LEARN_RATE=2e-5
export EPOCHS=40
export RE_DIR=<DATA PATH>/format_dt
export OUTPUT_DIR=<OUTPUT PATH>/bs-${BATCH_SIZE}_seqLen-${SEQ_LEN}_lr-${LEARN_RATE}_${EPOCHS}epoch_iter-$CASE_NUM
mkdir $OUTPUT_DIR
echo $OUTPUT_DIR

export TASK_NAME=bc7dp
export CUDA_VISIBLE_DEVICES=0 

python run_re_hfv4.py \
  --model_name_or_path ${LM_FULL_NAME} \
  --task_name $TASK_NAME \
  --do_train --do_eval --do_predict \
  --train_file $RE_DIR/train.tsv --validation_file $RE_DIR/dev.tsv --test_file $RE_DIR/test.tsv \
  --typeDict_file $RE_DIR/typeDict.json --vocab_add_file $RE_DIR/vocab_add.txt \
  --max_seq_length $SEQ_LEN \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size 512 \
  --learning_rate ${LEARN_RATE} \
  --num_train_epochs ${EPOCHS} --warmup_ratio 0.1 \
  --output_dir $OUTPUT_DIR/ \
  --logging_steps 2000 --eval_steps 2000 --save_steps 10000 \
  --seed $CASE_NUM
```

If you want to transform the formats of output predictions into the input format of DrugProt eval library, you can use our `transform_reTorch2bc7dp.py`.
```bash
python transform_reTorch2bc7dp.py --task=bc7dp \
 --output_path=$OUTPUT_DIR/predict_results_bc7dp.txt \
 --bc7format_out_path=$OUTPUT_DIR/pred_relations.tsv \
 --mapping_path=$RE_DIR/test-mapping.tsv \
 --label_path=$RE_DIR/typeDict.json
```
This will generate a transformed output file in `$OUTPUT_DIR/pred_relations.tsv `.
Use it to evaluate your trained model. 

The following bash scripts are an example of evaluating the model predictions on developement dataset (Please read "NOTE" of the "Pre-processed datasets" section). Please download development dataset from the DrugProt official website and unzip it on `DEV_DIR`.
```bash
export DRUGPROT_EVAL_LIB=${HOME}/github
export DEV_DIR=<PATH TO DEV FILES>/drugprot-gs-training-development/development

python ${DRUGPROT_EVAL_LIB}/drugprot-evaluation-library/src/main.py \
 -g $DEV_DIR/drugprot_development_relations.tsv \
 -p $OUTPUT_DIR/pred_relations.tsv \
 -e $DEV_DIR/drugprot_development_entities.tsv \
 --pmids $DEV_DIR/pmids.txt 2>&1 | tee -a $OUTPUT_DIR/BC7DP_pyTorch_score_2021Aug_total.log
```
<hr>

### For predicted large track data 
Automatic predictions of Drug-Protein relations database: Please check [here](http://wonjin.info/biore-yoon-et-al-2022/)

<hr>

## Citation info

Our main paper, entitled `Biomedical relation extraction with knowledge base refined weak-supervision` is under the review process of DATABASE journal - BioCreative special issues. 

Until our main paper is available as a journal article, please cite our short technical description, which is accepted and included in the BioCreative VII workshop proceedings.  

```
@inproceedings{yoon2021using,
  title={Using knowledge base to refine data augmentation for biomedical relation extraction},
  author={Yoon, Wonjin and Yi, Sean and Jackson, Richard and Kim, Hyunjae and Kim, Sunkyu and Kang, Jaewoo},
  booktitle={Proceedings of the BioCreative VII challenge evaluation workshop},
  pages={31--35},
  year={2021}
}
```
We will update citation info shortly.

<hr>

For inquiries, please contact `wjyoon (_at_) korea.ac.kr`.
