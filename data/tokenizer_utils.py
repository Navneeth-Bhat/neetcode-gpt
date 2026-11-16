from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        result=[]
        for n in numbers:
            string=str(n)
            i=0
            tokens=[]
            while i < len(string):
                j = i + 1
                longest = None

                while j <= len(string):
                    candidate = string[i:j]

                    if candidate in vocab:
                        longest = candidate
                    j += 1
                tokens.append(longest)
                i+=len(longest)
            result.append(tokens)
        return result

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        i=0
        count=0
        while i<len(text):
            j=i+1
            longest=None
            while j <= len(text):
                candidate = text[i:j]

                if candidate in vocab:
                    longest = candidate
                j += 1
            count+=1
            i+=len(longest)
        return count

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words=text.split()
        if not words:
            return 0.0
        count= self.count_tokens(text,vocab)
        return round(count/len(words),4)
