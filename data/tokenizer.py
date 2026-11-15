from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        chars=list(corpus)
        merges=[]
        for step in range(num_merges):
            if len(chars)<2:
                break
            freq={}
            for i in range(len(chars)-1):
                pair=(chars[i],chars[i+1])
                if pair in freq:
                    freq[pair]+=1
                else:
                    freq[pair]=1
            max_freq=max(freq.values())
            candidates=[pair for pair,count in freq.items() if count==max_freq]
            best_pair=min(candidates)
            i=0
            merges.append(list(best_pair))
            while i<len(chars)-1:
                pair=(chars[i],chars[i+1])
                if pair == best_pair:
                    chars[i:i+2]=[chars[i] + chars[i+1]]
                    i+=1
                else:
                    i+=1
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        return merges
