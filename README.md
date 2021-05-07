# JudicialYuan_POC_F2L
## crawler:
law_crawler_se.py : use selenium crawl data 

law_crawler.py :  use request crawl data (more accurate) 

dataprepare.py : build training data set  

prexplain_dataprepare : prepare json for Explainable AI

## train:
#### usge:
python train.py -n DistilBERT --train_record_file=./data/train.npz --batch_size=24 --lr_1=2e-5 --use_img=False

