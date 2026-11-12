import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        vocab=[]
        for sentence in positive+negative:
            vocab.extend(sentence.split())
        vocab=sorted((set(vocab)))
        ids={}
        
        for i in range(len(vocab)):
            ids[vocab[i]]=i+1
        
        positive_encoded=[]
        
        for sentence in positive:
            positive_encoded.append([ids[word] for word in sentence.split()])
        negative_encoded=[]
        
        for sentence in negative:
            negative_encoded.append([ids[word] for word in sentence.split()])    
        positive_tensors = [torch.tensor(seq, dtype=torch.float32) for seq in positive_encoded]                  
        negative_tensors = [torch.tensor(seq, dtype=torch.float32) for seq in negative_encoded]
        encoded=positive_tensors+negative_tensors
        return torch.nn.utils.rnn.pad_sequence(encoded,padding_value=0,batch_first=True)

