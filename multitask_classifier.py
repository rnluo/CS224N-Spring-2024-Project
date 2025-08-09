'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import datetime

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask

from itertools import permutations

TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('./bert-base-uncased-local') # local model
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### TODO

        self.num_labels = 5 # 5 sentiment classes
        self.sentiment_classifier = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.paraphrase_classifier = torch.nn.Linear(2 * config.hidden_size, 1)
        self.similarity_classifier = torch.nn.Linear(2 * config.hidden_size, 1)
        
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        #raise NotImplementedError


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO

        pooler_output = self.bert(input_ids, attention_mask)['pooler_output']

        return self.dropout(pooler_output)

        raise NotImplementedError


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO

        embedding = self.forward(input_ids, attention_mask)

        return self.sentiment_classifier(embedding)

        raise NotImplementedError


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO

        embedding_1 = self.forward(input_ids_1, attention_mask_1)
        embedding_2 = self.forward(input_ids_2, attention_mask_2)
        embedding_cat = torch.cat((embedding_1, embedding_2), dim=1)

        return self.paraphrase_classifier(embedding_cat)

        raise NotImplementedError


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO

        embedding_1 = self.forward(input_ids_1, attention_mask_1)
        embedding_2 = self.forward(input_ids_2, attention_mask_2)
        embedding_cat = torch.cat((embedding_1, embedding_2), dim=1)

        return self.similarity_classifier(embedding_cat)
    
        raise NotImplementedError
    
    def predict_cosine_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs the cosine embedding loss.
        '''
        ### TODO

        embedding_1 = self.forward(input_ids_1, attention_mask_1)
        embedding_2 = self.forward(input_ids_2, attention_mask_2)

        return F.cosine_similarity(embedding_1, embedding_2, dim=1)

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    writer = SummaryWriter(f'tf-logs/{datetime.datetime.now().strftime('%m/%d, %H%M%S') + args.filepath.replace(".pt", "")}')
    
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, _, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=para_dev_data.collate_fn)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model.args = args
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        sst_iter = iter(sst_train_dataloader)
        para_iter = iter(para_train_dataloader)
        sts_iter = iter(sts_train_dataloader)

        for _ in tqdm(range(len(para_train_dataloader)), desc=f'train-{epoch}', disable=TQDM_DISABLE):
            global_step = epoch * len(para_train_dataloader) + _

            optimizer.zero_grad()

            try:
                sst_batch = next(sst_iter)
            except StopIteration:
                sst_iter = iter(sst_train_dataloader)
                sst_batch = next(sst_iter)

            sst_ids, sst_mask, sst_labels = (sst_batch['token_ids'].to(device),
                                    sst_batch['attention_mask'].to(device), sst_batch['labels'].to(device))

            sst_logits = model.predict_sentiment(sst_ids, sst_mask)
            sst_loss = F.cross_entropy(sst_logits, sst_labels.view(-1), reduction='sum') / args.batch_size
        
            para_batch = next(para_iter)

            para_ids_1, para_mask_1, para_ids_2, para_mask_2, para_labels = (para_batch['token_ids_1'].to(device),
                                    para_batch['attention_mask_1'].to(device), para_batch['token_ids_2'].to(device),
                                    para_batch['attention_mask_2'].to(device), para_batch['labels'].to(device))

            para_logits = model.predict_paraphrase(para_ids_1, para_mask_1, para_ids_2, para_mask_2)
            para_loss = F.binary_cross_entropy_with_logits(para_logits.squeeze(), para_labels.float().squeeze(), reduction='sum') / args.batch_size
            

            try:
                sts_batch = next(sts_iter)
            except StopIteration:
                sts_iter = iter(sts_train_dataloader)
                sts_batch = next(sts_iter)

            sts_ids_1, sts_mask_1, sts_ids_2, sts_mask_2, sts_labels = (sts_batch['token_ids_1'].to(device),
                                    sts_batch['attention_mask_1'].to(device), sts_batch['token_ids_2'].to(device),
                                    sts_batch['attention_mask_2'].to(device), sts_batch['labels'].to(device))
            
            # Simple cosine similarity
            if args.cos_sim:
                sts_similarity = model.predict_cosine_similarity(sts_ids_1, sts_mask_1, sts_ids_2, sts_mask_2)
                sts_loss = F.mse_loss(sts_similarity, sts_labels.float() / 5)
            else:
                sts_logits = model.predict_similarity(sts_ids_1, sts_mask_1, sts_ids_2, sts_mask_2)
                sts_loss = F.mse_loss(sts_logits.squeeze(), sts_labels.float().squeeze(), reduction='sum') / args.batch_size
            
            # PCGrad implementation
            if args.pcgrad:
                optimizer.zero_grad()
                sst_loss.backward(retain_graph=True)
                sst_grad = [p.grad.clone() for p in model.bert.parameters() if p.grad is not None]

                optimizer.zero_grad()
                para_loss.backward(retain_graph=True)
                para_grad = [p.grad.clone() for p in model.bert.parameters() if p.grad is not None]

                optimizer.zero_grad()
                sts_loss.backward()
                sts_grad = [p.grad.clone() for p in model.bert.parameters() if p.grad is not None]
                
                grads = {'sst': sst_grad, 'para': para_grad, 'sts': sts_grad}

                for task_i, task_j in permutations(grads.keys(), 2):
                    g_i, g_j = grads[task_i], grads[task_j]
                    dot_product = sum(torch.sum(g_ik * g_jk) for g_ik, g_jk in zip(g_i, g_j))
        
                    if dot_product < 0:
                        scale = dot_product / sum(torch.sum(g_jk * g_jk) for g_jk in g_j)
                        grads[task_i] = [g_ik - scale * g_jk for g_ik, g_jk in zip(g_i, g_j)]

                optimizer.zero_grad()
                for k, p in enumerate(model.bert.parameters()):
                    if p.grad != None:
                        p.grad = grads['sst'][k] + grads['para'][k] + grads['sts'][k]
            else:
                loss = sst_loss + para_loss + sts_loss
                loss.backward()

            loss = sst_loss + para_loss + sts_loss

            writer.add_scalar('loss/train_total', loss, global_step)
            writer.add_scalar('loss/train_sst', sst_loss, global_step)
            writer.add_scalar('loss/train_para', para_loss, global_step)
            writer.add_scalar('loss/train_sts', sts_loss, global_step)

            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
        
        train_loss = train_loss / (num_batches)

        writer.add_scalar('loss/train_epoch', train_loss, epoch)

        # train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        # dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        sentiment_accuracy, _, _, paraphrase_accuracy, _, _, sts_corr, _, _ \
            = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)
        
        writer.add_scalar('accuracy/dev_sentiment', sentiment_accuracy, epoch)
        writer.add_scalar('accuracy/dev_paraphrase', paraphrase_accuracy, epoch)
        writer.add_scalar('correlation/dev_sts', sts_corr, epoch)

        if paraphrase_accuracy > best_dev_acc:
           best_dev_acc = paraphrase_accuracy
           save_model(model, optimizer, args, config, args.filepath)

        #print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}")

    writer.close()

def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model.args = args
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='train')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", type=bool, default=True)

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    # Custom arguments
    #parser.add_argument("--multitask_alternate", type=bool, help="Train alternately among the tasks", default=True)
    parser.add_argument("--pcgrad", type=bool, default=False)
    parser.add_argument("--cos_sim", type=bool, default=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)
