# JudicialYuan
## crawler:
law_crawler_se.py : use selenium crawl data 

law_crawler.py :  use request crawl data (more accurate than law_crawler_se.py) 

dataprepare.py : build training data set  

prexplain_dataprepare : prepare json for Explainable AI

## train:
#### usge:
python train.py -n DistilBERT --train_record_file=./data/train.npz --batch_size=24 --lr_1=2e-5

