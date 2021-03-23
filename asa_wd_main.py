import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy as np

from pytorch_transformers import BertModel

from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import Tokenizer4Bert, ABSADataset

from kvmn import BertKVMN


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))



class Instructor:
    def __init__(self, opt):
        self.opt = opt
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.bert_model)
        bert = BertModel.from_pretrained(opt.bert_model)
        self.model = BertKVMN(bert, num_labels=opt.polarities_dim, bert_dropout=opt.bert_dropout,
                                  feature_vocab_size=opt.feature_vocab_size, incro=opt.incro)
        self.model.to(opt.device)
        self.tokenizer = tokenizer

        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer, self.opt)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer, self.opt)

        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            torch.nn.init.xavier_uniform_(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def save_model(self, save_path, model, args):
        CONFIG_NAME = 'config.json'
        WEIGHTS_NAME = 'pytorch_model.bin'
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_path, WEIGHTS_NAME)
        output_config_file = os.path.join(save_path, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        with open(output_config_file, "w", encoding='utf-8') as writer:
            writer.write(model_to_save.config.to_json_string())
        output_args_file = os.path.join(save_path, 'training_args.bin')
        torch.save(args, output_args_file)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader, test_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None

        opt = self.opt
        results = {"bert_model": opt.bert_model, "dataset": opt.dataset, "batch_size": opt.batch_size,
                   "learning_rate": opt.learning_rate, "seed": opt.seed, "model_name": opt.model_name,
                   "l2reg": opt.l2reg, "incro": opt.incro, "mode": opt.mode}
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            # start_time = (datetime.now())
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]

                outputs = self.model(inputs)

                targets = sample_batched['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()

                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('epoch: {}, loss: {:.4f}, acc: {:.4f}'.format(epoch, train_loss, train_acc))
            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('>epoch: {}, val_acc: {:.4f}, val_f1: {:.4f}'.format(epoch, val_acc, val_f1))
            results["{}_val_acc".format(epoch)] = val_acc
            results["{}_val_f1".format(epoch)] = val_f1

            if val_acc > max_val_acc:
                max_val_acc = val_acc
                saving_path = os.path.join(self.opt.outdir, "epoch_{}".format(epoch))
                if not os.path.exists(saving_path):
                    os.makedirs(saving_path)
                self.save_model(saving_path, self.model, self.opt)

                self.model.eval()
                saving_path = os.path.join(self.opt.outdir, "epoch_{}_eval.txt".format(epoch))
                test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader, saving_path)
                logger.info('>> epoch: {}, test_acc: {:.4f}, test_f1: {:.4f}'.format(epoch, test_acc, test_f1))
                print(('>> epoch: {}, test_acc: {:.4f}, test_f1: {:.4f}'.format(epoch, test_acc, test_f1)))

                results["max_val_acc"] = max_val_acc
                results["test_acc"] = test_acc
                results["test_f1"] = test_f1


            output_eval_file = os.path.join(self.opt.outdir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                # writer.write(json.dumps(results, ensure_ascii=False))
                for k,v in results.items():
                    writer.write("{}={}\n".format(k,v))
        return path

    def _evaluate_acc_f1(self, data_loader, saving_path=None):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()

        saving_path_f = open(saving_path, 'w') if saving_path is not None else None
 
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):

                cols = self.opt.inputs_cols

                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_raw_texts = t_sample_batched['raw_text']
                t_aspects = t_sample_batched['aspect']

                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

                if saving_path_f is not None:
                    for t_target, t_output, t_raw_text, t_aspect in zip(t_targets.detach().cpu().numpy(),
                                                                        torch.argmax(t_outputs, -1).detach().cpu().numpy(),
                                                                        t_raw_texts, t_aspects):
                        saving_path_f.write("{}\t{}\t{}\t{}\n".format(t_target, t_output, t_raw_text, t_aspect))
        
        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        self._train(criterion, optimizer, train_data_loader, val_data_loader, test_data_loader)



def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_kv_aspect-second', type=str)
    parser.add_argument('--dataset', default='demo', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default='2e-5', type=float)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--bert_dropout', default=0.2, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=10, type=int)#15
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--max_seq_len', default=100, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=40, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0.1, type=float, help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--knowledge_type', default='dep', type=str)
    parser.add_argument('--log', default='log', type=str)
    parser.add_argument('--bert_model', default='./bert-large-uncased', type=str)
    parser.add_argument('--f_t', default=3, type=int)
    parser.add_argument('--incro', default='cat', type=str)
    parser.add_argument('--mode',default='case', type=str)
    parser.add_argument('--conflict', default=0, type=int)
    parser.add_argument('--abalation', default='none', type=str)
    parser.add_argument('--outdir', default='./', type=str)
    opt = parser.parse_args()

    if not os.path.exists(opt.outdir):
        try:
            os.mkdir(opt.outdir)
        except Exception as e:
            print(str(e))
    import datetime
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    opt.outdir = os.path.join(opt.outdir, "{}_bts_{}_lr_{}_l2reg_{}_seed_{}_bert_dropout_{}_{}".format(
        opt.dataset,
        opt.batch_size,
        opt.learning_rate,
        opt.l2reg,
        opt.seed,
        opt.bert_dropout,
        now_time
    ))
    if not os.path.exists(opt.outdir):
        os.mkdir(opt.outdir)

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = '{}/{}-{}-{}.log'.format(opt.log, opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()



if __name__ == '__main__':
    main()
