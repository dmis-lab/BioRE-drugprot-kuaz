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

### How to train the model / make predictions using trained model
First, train your model using `run_re_hfv4.py`.

For example, the following code will produce 
```bash
export $SEED=0
export CASE_NUM=`printf %02d $SEED`

export LM_FULL_NAME=<LM PATH or HF Transformer name/url>
export SEQ_LEN=192
export BATCH_SIZE=16 #16 with LR 2e-5  #32 with LR 5e-5
export LEARN_RATE=2e-5
export EPOCHS=40
export RE_DIR=<DATA PATH>/format_dt
export OUTPUT_DIR=<OUTPUT PATH>/bs-${BATCH_SIZE}_seqLen-${SEQ_LEN}_lr-${LEARN_RATE}_${EPOCHS}epoch_iter-$CASE_NUM
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
Prediction results and checkpoints in `$OUTPUT_DIR`.

If you want to transform the formats of output predictions into the input format of DrugProt eval library, you can use our `transform_reTorch2bc7dp.py`.
```bash
export DEV_DIR=<PATH TO DEV FILES>/drugprot-gs-training-development/development

python transform_reTorch2bc7dp.py --task=bc7dp \
 --output_path=$OUTPUT_DIR/predict_results_bc7dp.txt \
 --bc7format_out_path=$OUTPUT_DIR/pred_relations.tsv \
 --mapping_path=$RE_DIR/test-mapping.tsv \
 --label_path=$RE_DIR/typeDict.json
```
This will generate a transformed output file in `$OUTPUT_DIR/pred_relations.tsv `.

Use it to evaluate your trained model.
```bash
export DRUGPROT_EVAL_LIB=${HOME}/github

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

For inquiries, please contact `wjyoon (_at_) korea.ac.kr`.