from collections import Counter

class ChrTokenizer():
    def __init__(self, text='', vocabThreshold=0, charset="", pad="[PAD]", unk="[UNK]", mask=None, vocab=None):
        if vocab==None:
            charsCnt = Counter(list('\n'.join(text)))
            vocab=[tok for tok, count in charsCnt.items() if count >= vocabThreshold and (tok in charset)]
            if mask!=None:
                vocab = [pad]+sorted(vocab, key=lambda x: charsCnt[x], reverse=True)+[mask]+[unk]
            else:
                vocab = [pad]+sorted(vocab, key=lambda x: charsCnt[x], reverse=True)+[unk]
            

        self.vocab = list(vocab)
        self.vocab_lookup = {char: idx for idx, char in enumerate(vocab)}
        self.vocab_size = len(self.vocab)
        self.pad_str = vocab[0]
        self.pad_tok = 0
        self.unk_str = vocab[-1]
        self.unk_tok = self.vocab_size-1
        if mask!=None:
            self.unk_str = vocab[-2]
            self.mask_tok = self.vocab_size-2
        
    def encode(self, text, max_length=0, language = None):
        if language == 'HUN':
            s = [self.char_transforms_hun(char) for char in list(text)]
        else:
            s = [self.vocab_lookup[char] if char in self.vocab_lookup 
                else self.vocab_lookup[self.unk_str]
                for char in list(text)]
                              
        if max_length==0 or max_length>len(s):
            max_length=len(s)     
        
        return s[:max_length]
    
    def decode(self, seq, orig_str=None, language = None):
        s=''
        if language == 'HUN':
            assert(orig_str!=None)
            for token,char in zip(seq,list(orig_str)):
                s += self.char_transforms_decode_hun(token,char)
        else:
            if orig_str==None:
                for token in seq:
                    c = self.vocab[token]
                    s += c
            else:
                for idx,token in enumerate(seq):
                    c = self.vocab[token]
                    if c==self.unk_str:
                        c = orig_str[idx]
                    s += c
        return s
    
    def char_transforms_pol(self,char):
        char=char.lower()
        if char in ['a','c','e','l','n','o','s','z']:
            return 1
        if char in ['ć','ń','ó','ś','ź']:
            return 2
        if char in ['ą','ę']:
            return 3
        if char in ['ł']:
            return 4
        if char in ['ż']:
            return 5
        return 0

    def char_transforms_hun(self,char):
        char=char.lower()
        if char in ['a','e','i','o','u']:
            return 1
        if char in ['á','é','í','ó','ú']:
            return 2
        if char in ['ö','ü']:
            return 3
        if char in ['ő','ű']:
            return 4
        return 0

    def ct_decode_rule(self,tok,char,strict,char_list=['a','á']):
        for i in range(4):
            if len(char_list)<=i:
                break
            if tok==i+1:
                return self.copy_case(char,char_list[i])
        if strict:
            return '_'
        return self.copy_case(char,char_list[0])

    def char_transforms_decode_hun(self,tok,char,strict=True):
        if char.lower() in ['a','á']:
            return self.ct_decode_rule(tok,char,strict,['a','á'])
        if char.lower() in ['e','é']:
            return self.ct_decode_rule(tok,char,strict,['e','é'])   
        if char.lower() in ['i','í']:
            return self.ct_decode_rule(tok,char,strict,['i','í'])
        if char.lower() in ['o','ó','ö','ő']:
            return self.ct_decode_rule(tok,char,strict,['o','ó','ö','ő'])
        if char.lower() in ['u','ú','ü','ű']:
            return self.ct_decode_rule(tok,char,strict,['u','ú','ü','ű'])
        if tok==0:
            return char
        if strict:
            return '_'
        return char

    def copy_case(self,char_from, char_to):
        if char_from.isupper():
            return char_to.upper()
        return char_to.lower()