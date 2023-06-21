
[Official Doc](https://github.com/PaddlePaddle/PaddleOCR)

![[PaddleOCR ARC.png]]

### PP-OCR system for the optical character recognition system.

- [Part I: Review overall architecture and text detector on paper](https://medium.com/@anhtuan_40207/review-paper-pp-ocr-a-practical-ultra-lightweight-ocr-system-part-i-b5d2fcfe74cf)
-   [Part II: Review the direction classification model and text recognitor](https://medium.com/@anhtuan_40207/paper-review-pp-ocr-a-practical-ultra-lightweight-ocr-system-part-ii-b67a5e511b2)

# 1. Installation

Firstly, install the official code from GitHub:

```
git clone https://github.com/PaddlePaddle/PaddleOCR.git
```

If you have only CPU, run:

```
pip install paddleocr --upgrade
python3 -m pip install paddlepaddle
```

Then, I install the pretrained model. 
[You can find it here:](https://github.com/PaddlePaddle/PaddleOCR#%EF%B8%8F-pp-ocr-series-model-listupdate-on-september-8th)
#### [OCR Model List](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_en/models_list_en.md#23-multilingual-recognition-modelupdating)

# 2. Inference

For example, I use the English ultra-lightweight PP-OCRv3 model for inference. I download inference models for detection, direction classification, and recognition and save them to /inference/det, /inference/cls/, /inference/reg respectively and extract them.
[Python Inference](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/docs/inference_en.md)
[Python Inference for PP-OCR Model Zoo](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/inference_ppocr_en.md)


```
cd PaddleOCR/ppstructure 

# download model 
mkdir inference 

cd inference 

# Download the detection model of the ultra-lightweight table English OCR model and unzip it 
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar && tar xf en_ppocr_mobile_v2.0_table_det_infer.tar 

# Download the recognition model of the ultra-lightweight table English OCR model and unzip it 
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar && tar xf en_ppocr_mobile_v2.0_table_rec_infer.tar 

# Download the ultra-lightweight English table inch model and unzip it wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar && tar xf en_ppocr_mobile_v2.0_table_structure_infer.tar 

##New OCR Model 
#!wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar && tar xf en_PP-OCRv3_det_infer.tar
```

Run file:

```
PaddleOCR/ppstructure 

python3 /path/PaddleOCR/ppstructure/table/predict_table.py --det_model_dir=inference/en_PP-OCRv3_det_infer --rec_model_dir=inference/en_ppocr_mobile_v2.0_table_rec_infer --table_model_dir=inference/en_ppocr_mobile_v2.0_table_structure_infer --image_dir=/content/PaddleOCR/ppstructure/table_2.png --rec_char_dict_path=../ppocr/utils/dict/table_dict.txt --table_char_dict_path=../ppocr/utils/dict/table_structure_dict.txt --det_limit_side_len=736 --det_limit_type=min --output ./output/table
```

# 3. Training

For training, You can see details in the authors’s source:

-   Training text detection:
	[PaddleOCR/detection_en.md at release/2.5 · PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_en/detection_en.md#2-training)

-   Traing Text Direction Classification:
	[PaddleOCR/angle_class_en.md at release/2.5 · PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_en/angle_class_en.md)

-   Training text recoginition:
	[PaddleOCR/recognition_en.md at release/2.5 · PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_en/recognition_en.md)

