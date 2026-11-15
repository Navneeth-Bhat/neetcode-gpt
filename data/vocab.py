from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        stoi={}
        itos={}
        text=sorted(list(set(text)))
        for i,t in enumerate(text):
            stoi[t]=i
            itos[i]=t
        return (stoi,itos)

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        converted=[]
        for t in text:
            converted.append(stoi[t])
        return converted

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        decoded=[]
        for i in ids:
            decoded.append(itos[i])
        return "".join(decoded)
