"""
@author: Jehad Aldahdooh
"""
import torch.nn as nn
from transformers import AutoModel

class DTI_Model(nn.Module):
    def __init__(self,model_name):
        super(DTI_Model, self).__init__()
        self.model = AutoModel.from_pretrained(model_name) 
        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 2) 
        self.dropout = nn.Dropout(0.1)

    def forward(self,input_id_texts, attention_masks_texts=None):
        sequence_output, _ = self.model(   
                input_ids=input_id_texts,
                attention_mask=attention_masks_texts,
                
        )
        x=self.dropout(sequence_output[:,0,:])        
        linear_layer = self.linear1(x.view(-1,768)) 
        logits = self.linear2(linear_layer)
        return logits
