# SCATTER: Selective Context Attentional Scene Text Recognizer

CVPR2020 OCR SCATTER 复现

Based on [What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis](https://arxiv.org/abs/1904.01906)

SCATTER: Selective Context Attentional Scene Text Recognizer | [paper](https://arxiv.org/abs/2003.11288)

### Train
```python
python train_SCATTER.py \
  --exp_name SCATTER \
  --train_data train_data_path \
  --valid_data valid_data_path \ 
  --batch_size 64 \
  --select_data MJ-ST \ 
  --rgb \ 
  --Transformation None \
  --FeatureExtraction None \
  --SequenceModeling None \
  --Prediction None \
  --LSTM_Layer 2 \
  --Selective_Layer 1
  --fp16(optional)
```
### Test
```python
python test_SCATTER.py \
  --eval_data /imdb/evaluation/data/path \
  --benchmark_all_eval \
  --batch_max_length 100 \
  --batch_size 128 \
  --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None \
  --LSTM_Layer 2 --Selective_Layer 4 \
  --saved_model /pretrained/model/path
```
