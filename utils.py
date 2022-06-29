import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, AdamW
import time, datetime, random, re
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModel,
)
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import gcf

def seed_all(seed_value):
    random.seed(seed_value) 
    np.random.seed(seed_value)
    torch.manual_seed(seed_value) 
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) 
        torch.backends.cudnn.deterministic = True 
        torch.cuda.amp.autocast(enabled=True)
        torch.backends.cudnn.benchmark = False

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# plot results
def plot_results(df):
    # styling from seaborn.
    sns.set(style='darkgrid')
    # uncrease the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # plot the learning curve.
    plt.plot(df_stats['Train Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Val Loss'], 'g-o', label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(list(range(1, epochs+1)))
    return plt.show()


def clean(text):
    text = text.split()
    text = [x.strip() for x in text] #strip remove the whitespces in the text
    text = [x.replace('\n', ' ').replace('\t', ' ') for x in text]
    text = ' '.join(text)
    return text
    

# prepare data
def clean_df(df):
    df['body'] = df['body'].apply(clean)
    # lower case the data
    df['body'] = df['body'].apply(lambda x: x.lower())
    # remove excess spaces near punctuation
    df['body'] = df['body'].apply(lambda x: re.sub(r'\s([?.!"](?:\s|$))', r'\1', x))
    # remove excess white spaces
    df['body'] = df['body'].apply(lambda x: " ".join(x.split()))
    return df


class DTDataset(Dataset):
    def __init__(self, df, indices,model_name, set_type=None):
        super(DTDataset, self).__init__()

        df = df.iloc[indices]
        self.alldata = df['body'].values.tolist()
        self.pmid=df['pmid'].values.tolist()
        self.set_type = set_type
        if self.set_type != 'test':
            self.labels = df['target'].values

        self.max_length2 = 512 #512
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) #'allenai/scibert_scivocab_uncased'
        
    def __len__(self):
        return len(self.alldata)

    def __getitem__(self, index):
        #print(self.labels[index])
        #print(self.alldata[index])
        
        tokenized_titles = self.tokenizer.encode_plus(
                            self.alldata[index],  # document to encode.
                            add_special_tokens=True,  # add '[CLS]' and '[SEP]'
                            max_length=self.max_length2,  # set max length
                            truncation=True,  # truncate longer messages
                            pad_to_max_length=True,  # add padding
                            return_attention_mask=True,  # create attn. masks
                            return_token_type_ids=False,
                            return_tensors='pt'  # return pytorch tensors
                       )

        input_ids_titles = tokenized_titles['input_ids'].squeeze()
        attention_mask_titles = tokenized_titles['attention_mask'].squeeze()
        
        if self.set_type != 'test':
            return {
                'data': {
                    'input_ids': input_ids_titles.long(),
                    'attention_mask': attention_mask_titles.long(),
                },
                'labels': torch.tensor(self.labels[index]).int(), #torch.Tensor([self.labels[index]]).int(),
                #'real':self.pmid[index]
            }
        return {
            'data': {
                'input_ids': input_ids_titles.long(),
                'attention_mask': attention_mask_titles.long(),
            }
        }


def train(model, dataloader, optimizer):
    """
    Train a model for one epoch on the input data with a given optimizer and
    criterion.
    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        epoch_number: The number of the epoch for which training is performed.
    Returns:
        None
        - Prints the following:
            training loss | trainig f1 weighted | train f1 macro | train f1 micro | training time
    """
    # capture time
    total_t0 = time.time()
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    print('Training...')
    # reset total loss for epoch
    train_total_loss = 0
    total_train_f1_weighted = 0
    total_train_f1_macro = 0
    total_train_f1_micro = 0
    # put model into traning mode
    model.train()
    # for each batch of training data...
    for step, batch in enumerate(dataloader):
        # progress update every 600 batches.
        if step % 600 == 0 and not step == 0:
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))
        # Unpack this training batch from our dataloader:
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        #print('yes1')
        b_input_ids = batch['data']['input_ids'].cuda()
        b_input_mask = batch['data']['attention_mask'].cuda()
        b_labels = batch['labels'].cuda().long()
        # clear previously calculated gradients
        optimizer.zero_grad()

        # runs the forward pass with autocasting.
        with autocast():
            # forward propagation (evaluate model on training batch)
            logits = model(b_input_ids,b_input_mask,labels=b_labels)
            # sum the training loss over all batches for average loss at end
            # loss is a tensor containing a single value
            loss = criterion(logits, b_labels)
            train_total_loss += loss.item()
        
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        # update the learning rate
        scheduler.step()

        # move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        y_true = b_labels.detach().cpu().numpy()

        # calculate preds
        rounded_preds = np.argmax(logits, axis=1).flatten()

        
        # calculate different f1 scores
        total_train_f1_weighted += f1_score(rounded_preds, y_true,average='weighted') 
        total_train_f1_macro += f1_score(rounded_preds, y_true,average='macro')
        total_train_f1_micro += f1_score(rounded_preds, y_true,average='micro')

    # calculate the average loss over all of the batches
    avg_train_loss = train_total_loss / len(dataloader)

    # calculate the average f1 over all of the batches
    avg_train_f1_weighted = total_train_f1_weighted / len(dataloader)
    avg_train_f1_macro = total_train_f1_macro / len(dataloader)
    avg_train_f1_micro = total_train_f1_micro / len(dataloader)

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'Train Loss': avg_train_loss,
            'Train F1 weighted': avg_train_f1_weighted,
            'Train F1 macro': total_train_f1_macro,
            'Train F1 micro': total_train_f1_micro,
            
        }
    )

    # training time end
    training_time = format_time(time.time() - total_t0)

    # print result summaries
    print("")
    print("summary results")
    print("epoch | trn loss | trn f1 weighted | avg_train_f1_macro | avg_train_f1_micro | trn time ")
    print(f"{epoch+1:5d} | {avg_train_loss:.5f} | {avg_train_f1_weighted:.5f} | {avg_train_f1_macro:.5f} |{avg_train_f1_micro:.5f} |{training_time:}")

    torch.cuda.empty_cache()

    return None


def validating(model, dataloader):

    # capture validation time
    total_t0 = time.time()
    print("Running Validation...")
    # put the model in evaluation mode
    model.eval()

    # track variables
    total_valid_accuracy = 0
    total_valid_loss = 0
    total_valid_f1_weighted = 0
    total_valid_f1_macro = 0
    total_valid_f1_micro = 0
    total_valid_recall = 0
    total_valid_precision = 0


    # evaluate data for one epoch
    for batch in dataloader:
        # Unpack this training batch from our dataloader:
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch['data']['input_ids'].cuda()
        b_input_mask = batch['data']['attention_mask'].cuda()
        b_labels = batch['labels'].cuda().long()

        # tell pytorch not to bother calculating gradients
        # as its only necessary for training
        with torch.no_grad():

            # forward propagation (evaluate model on training batch)
            logits = model(b_input_ids,b_input_mask,labels=b_labels)
            loss = criterion(logits, b_labels)

        # accumulate validation loss
        total_valid_loss += loss.item()

        # move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        y_true = b_labels.detach().cpu().numpy()

        # calculate preds
        rounded_preds = np.argmax(logits, axis=1).flatten()

        # calculate f1
        total_valid_f1_weighted += f1_score(rounded_preds, y_true,
                                   average='weighted',
                                   labels=np.unique(rounded_preds))

        total_valid_f1_macro += f1_score(rounded_preds, y_true,
                                   average='macro',
                                   labels=np.unique(rounded_preds))
        total_valid_f1_micro += f1_score(rounded_preds, y_true,
                                   average='micro',
                                   labels=np.unique(rounded_preds))

        # calculate accuracy
        total_valid_accuracy += accuracy_score(rounded_preds, y_true)

        # calculate precision
        total_valid_precision += precision_score(rounded_preds, y_true,
                                                 average='weighted',
                                                 labels=np.unique(rounded_preds))

        # calculate recall
        total_valid_recall += recall_score(rounded_preds, y_true,
                                                 average='weighted',
                                                 labels=np.unique(rounded_preds))

    # report final accuracy of validation run
    avg_accuracy = total_valid_accuracy / len(dataloader)

    # report final f1 of validation run
    global avg_val_f1_weighted
    global avg_val_f1_macro
    global avg_val_f1_micro
    
    avg_val_f1_weighted = total_valid_f1_weighted / len(dataloader)
    avg_val_f1_macro = total_valid_f1_macro / len(dataloader)
    avg_val_f1_micro = total_valid_f1_micro / len(dataloader)

    # report final f1 of validation run
    avg_precision = total_valid_precision / len(dataloader)

    # report final f1 of validation run
    avg_recall = total_valid_recall / len(dataloader)

    # calculate the average loss over all of the batches.
    global avg_val_loss
    avg_val_loss = total_valid_loss / len(dataloader)

    # Record all statistics from this epoch.
    valid_stats.append(
        {
            'Val Loss': avg_val_loss,
            'Val Accur.': avg_accuracy,
            'Val precision': avg_precision,
            'Val recall': avg_recall,
            'Val F1 weighted': avg_val_f1_weighted,
            'Val F1 macro': avg_val_f1_macro,
            'Val F1 micro': avg_val_f1_micro
            
        }
    )

    # capture end validation time
    training_time = format_time(time.time() - total_t0)

    # print result summaries
    print("")
    print("summary results")
    print("epoch | val loss | val f1 weighted | val f1 macro | val f1 micro | val time")
    print(f"{epoch+1:5d} | {avg_val_loss:.5f} | {avg_val_f1_weighted:.5f} | {avg_val_f1_macro:.5f} |{avg_val_f1_micro:.5f} |{training_time:}")

    return None



def testing_pubtator(model, dataloader):
    print("Running Pubtator Testing...")
    test_pred = []
    model.eval()
    for batch in dataloader:
        b_input_ids = batch['data']['input_ids'].cuda()
        b_input_mask = batch['data']['attention_mask'].cuda()
        with torch.no_grad():
            logits = model(b_input_ids,
                                 b_input_mask)
        logits = logits.detach().cpu().numpy()
        rounded_preds = np.argmax(logits, axis=1).flatten()
        test_pred.extend(rounded_preds)
    return test_pred

