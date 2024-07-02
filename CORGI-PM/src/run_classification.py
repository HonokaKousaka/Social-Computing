import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch import cuda
from sklearn.metrics import classification_report


class CustomDataset(Dataset):
    def __init__(self, text_array, label_array, tokenizer, max_len):
        self.tokenizer = tokenizer
        
        
        self.max_len = max_len
        self.targets = label_array
        # self.encodings = self.tokenizer(text_array, truncation=True, padding=True)
        self.encodings = self.tokenizer(text_array, add_special_tokens=True,max_length=self.max_len,pad_to_max_length=True,return_token_type_ids=True)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        ids = self.encodings['input_ids'][index]
        mask = self.encodings['attention_mask'][index]
        token_type_ids = self.encodings["token_type_ids"][index]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)



def main():

    def train(epoch):
        model.train()
        for _, data in enumerate(training_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
            loss = loss_fn(outputs['logits'], targets)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if _%500==0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')


    def validation(data_to_test_loader):
        model.eval()
        fin_targets=[]
        fin_outputs=[]
        with torch.no_grad():
            for _, data in enumerate(data_to_test_loader, 0):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.float)
                outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)['logits']
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets

    task = 'detection'
    print(f'running baselines for task {task}')
    
    device = 'cuda' if cuda.is_available() else 'cpu'

    bias_corpus = np.load('dataset/CORGI-PC_splitted_biased_corpus_v1.npy',allow_pickle=True).item()
    non_bias_corpus = np.load('dataset/CORGI-PC_splitted_non-bias_corpus_v1.npy',allow_pickle=True).item()
    weibo_bias = np.load('dataset/bias_sentence.npy')
    weibo_unbias = np.load('dataset/unbias_sentence.npy')
    X_train = np.array( list(bias_corpus['train']['ori_sentence']) + non_bias_corpus['train']['text'] )
    y_train = np.array( [[0,1] for _ in list(bias_corpus['train']['ori_sentence'])] + [[1,0] for _ in non_bias_corpus['train']['text']] )
    X_valid = np.array( list(bias_corpus['valid']['ori_sentence']) + non_bias_corpus['valid']['text'] )
    y_valid = np.array( [[0,1] for _ in list(bias_corpus['valid']['ori_sentence'])] + [[1,0] for _ in non_bias_corpus['valid']['text']] )
    X_test = np.array( list(bias_corpus['test']['ori_sentence']) + non_bias_corpus['test']['text'] )
    y_test = np.array( [[0,1] for _ in list(bias_corpus['test']['ori_sentence'])] + [[1,0] for _ in non_bias_corpus['test']['text']] )
    X_real = np.array( list(weibo_bias) + list(weibo_unbias) )
    y_real = np.array( [[0,1] for _ in list(weibo_bias)] + [[1,0] for _ in list(weibo_unbias)] )
        
    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape, X_real.shape, y_real.shape)

    
    # define parameters
    model_list = ['hfl/chinese-electra-180g-base-discriminator', 'hfl/chinese-bert-wwm-ext', 'hfl/chinese-xlnet-base']
    lr_list = [1e-5,  5e-6, 1e-5]
    # Defining some key variables that will be used later on in the training
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 32
    EPOCHS = 1
    
    for i in [0,1,2]:
        model_name = model_list[i]
        print('########################################')

        print('start training', model_name)


        LEARNING_RATE = lr_list[i] # 1e-4

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        training_set = CustomDataset(list(X_train), y_train, tokenizer, MAX_LEN)
        validation_set = CustomDataset(list(X_valid), y_valid, tokenizer, MAX_LEN)
        testing_set = CustomDataset(list(X_test), y_test, tokenizer, MAX_LEN)
        real_set = CustomDataset(list(X_real), y_real, tokenizer, MAX_LEN)

        train_params = {'batch_size': TRAIN_BATCH_SIZE,
                        'shuffle': True,
                        'num_workers': 0
                        }

        test_params = {'batch_size': VALID_BATCH_SIZE,
                        'shuffle': True,
                        'num_workers': 0
                        }
        
        real_params = {'batch_size': VALID_BATCH_SIZE,
                       'shuffle': True,
                       'num_workers': 0
                        }

        training_loader = DataLoader(training_set, **train_params)
        validation_loader = DataLoader(validation_set, **test_params)
        testing_loader = DataLoader(testing_set, **test_params)
        real_loader = DataLoader(real_set, **real_params)

        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2,).to(device)
        
        print(model.config)
        print(model)


        optimizer = torch.optim.AdamW(params =  model.parameters(), lr=LEARNING_RATE)


        num_training_steps = EPOCHS * len(training_loader)

        lr_scheduler = transformers.get_scheduler(

            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps

        )


        print(f'training {model_name}')
        for epoch in range(EPOCHS):
            train(epoch)


        for mode, loader in zip(['valid','test','train','real'], [validation_loader, testing_loader, training_loader, real_loader]):
            print(f'----------------Evaluating for {mode} set----------------')
            outputs, targets = validation(loader)
            outputs = np.array(outputs) >= 0.5
            print(classification_report(y_true=targets, y_pred=outputs,digits=6))

            accuracy = metrics.accuracy_score(targets, outputs)
            f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
            f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
            print(f"Accuracy Score = {accuracy}")
            print(f"F1 Score (Micro) = {f1_score_micro}")
            print(f"F1 Score (Macro) = {f1_score_macro}")
    
            
        
        del model
        torch.cuda.empty_cache()
            
if __name__ == "__main__":
    main()