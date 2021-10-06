import argparse 

def get_crawler_args():
    """Get arguments needed in law_crawler.py"""
    parser = argparse.ArgumentParser('training data cfact rawler')

    parser.add_argument('--input_file',
                        type=str,
                        default='./data/output1.xlsx',
                        help='File to load for the scraped laws')

    parser.add_argument('--ouput_file_path',
                        type=str,
                        default='./data/crawler_result.json',
                        help='File to save for the scraped laws')

    args = parser.parse_args()

    return args

def get_dataprepare_args():
    """Get arguments needed in dataprepare.py"""
    parser = argparse.ArgumentParser('data processing for training')

    add_common_args(parser) 

    parser.add_argument('--raw_train_data',
                        type=str,
                        default='./data/crawler_result.json',
                        help='File to load for raw train processing')
    
    parser.add_argument('--uid_raws',
                        type=str,      
                        default='./data/jud_drug_nsd00_401_1.csv',
                        help='File to load for raw train processing')

    parser.add_argument('--testdata',
                        type=str,
                        default='./data/all_done.json',
                        help='File to load for raw train processing')

    parser.add_argument('--crawler_result',
                        type=str,
                        default='./data/crawler_result.json',
                        help='File to load for raw train processing')

    parser.add_argument('--train_file',
                        type=str,
                        default='./data/train.npz',
                        help='File to save for the traing data')

    parser.add_argument('--input_len',
                        type=int,
                        default=5000,
                        help='Length of the model input')

    parser.add_argument('--content_max_len',
                        type=int,
                        default=510,
                        help='Maximun length of tokenized content')

    parser.add_argument('--train_size',
                        type=float,
                        default=0.90,
                        help='Proportion of data in training set')

    args = parser.parse_args()

    return args

def get_train_args():
    """Get arguments needed in train.py"""
    parser = argparse.ArgumentParser('Train a model on NEWS')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--eval_steps',
                        type=int,
                        default=2000,
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--lr_1',
                        type=float,
                        default=2e-3,
                        help='Learning rate for classifier layer.')
    parser.add_argument('--lr_2',
                        type=float,
                        default=2e-5,
                        help='Fine tuned learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=0,
                        help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=20,
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.1,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--num_labels',
                        type=int,
                        default=4,
                        help='Number of labels for classification.')
    parser.add_argument('--freeze',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether to freeze the pretrained BERT model')
    parser.add_argument('--metric_name',
                        type=str,
                        default='F1',
                        choices=('NLL', 'F1'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')

    args = parser.parse_args()

    if args.metric_name == 'NLL':
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name == 'F1':
        # Best checkpoint is the one that maximizes F1
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')

    return args

def prexplain_dataprepare_args():
    """Get arguments needed in prexplain_dataprepare.py"""
    parser = argparse.ArgumentParser('for ai explainer')

    parser.add_argument('--input_file_uid_judge_clean',
                        type=str,
                        default='./data/crawler_result.json',
                        help='File to save for the scraped laws')

    parser.add_argument('--input_file',
                        type=str,
                        default='./data/output1.xlsx',
                        help='File to load for the Judicial data')

    parser.add_argument('--ouput_file',
                        type=str,
                        default='./data/prexplain.json',
                        help='File to save for the prexplain data')

    parser.add_argument('--uid_raws',
                        type=str,      
                        default='./data/jud_drug_nsd00_401_1.csv',
                        help='File to load for raw train processing')

    args = parser.parse_args()

    return args



def add_common_args(parser):
    """Add arguments common to scripts: dataprepare.py, train.py, test.py"""
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=False,
                        help='Name to identify training or test run.')
    parser.add_argument('--raw_data_file',
                        type=str,
                        default='./data/train_raw.npz',
                        help='Original data file')
    parser.add_argument('--times',
                        type=str,
                        default='liberty',
                        help='Determine publisher to process')
    parser.add_argument('--seed',
                        type=int,
                        default=112,
                        help='Random seed for reproducibility')
    parser.add_argument('--train_record_file',
                        type=str,
                        default='./data/train.npz')
    parser.add_argument('--dev_record_file',
                        type=str,
                        default='./data/dev.npz')
    parser.add_argument('--test_record_file',
                        type=str,
                        default='./data/test.npz')              