import random
import numpy as np

class Soft_deaccent():
    def __init__(self, lang, tokenizer):
        all_rules={
            "HU":{
                "é": ["e"],
                "á": ["a"],
                "í": ["i"],
                "ö": ["o"],
                "ü": ["u"],
                "ó": ["o"],
                "ő": ["o","ö"],
                "ú": ["u"],
                "ű": ["u","ü"]
            },
            "PL":{
                "ć": ["c"],
                "ń": ["n"],
                "ó": ["o"],
                "ś": ["s"],
                "ź": ["z"],
                "ą": ["a"],
                "ę": ["e"],
                "ł": ["l"],
                "ż": ["z"]
            },
            "CZ":{
                "á": ["a"],
                "č": ["c"],
                "ď": ["d"],
                "é": ["e"],
                "ě": ["e"],
                "í": ["i"],
                "ň": ["n"],
                "ó": ["o"],
                "ř": ["r"],
                "š": ["s"],
                "ť": ["t"],
                "ú": ["u"],
                "ů": ["u"],
                "ý": ["y"],
                "ž": ["z"]
            },
            "SK":{
                "á": ["a"],
                "ä": ["a"],
                "č": ["c"],
                "ď": ["d"],
                "é": ["e"],
                "í": ["i"],
                "ĺ": ["l"],
                "ľ": ["l"],
                "ň": ["n"],
                "ó": ["o"],
                "ô": ["o"],
                "ŕ": ["r"],
                "š": ["s"],
                "ť": ["t"],
                "ú": ["u"],
                "ý": ["y"],
                "ž": ["z"]
            },
			"VI":{
                "á": ["a"],
                "à": ["a"],
                "ả": ["a"],
                "ã": ["a"],
                "ạ": ["a"],
                "ă": ["a"],
                "ắ": ["a","ă"],
                "ằ": ["a","ă"],
                "ẳ": ["a","ă"],
                "ẵ": ["a","ă"],
                "ặ": ["a","ă"],
                "â": ["a"],
                "ấ": ["a","â"],
                "ầ": ["a","â"],
                "ẩ": ["a","â"],
                "ẫ": ["a","â"],
                "ậ": ["a","â"],
                "é": ["e"],
                "è": ["e"],
                "ẻ": ["e"],
                "ẽ": ["e"],
                "ẹ": ["e"],
                "ê": ["e"],
                "ế": ["e","ê"],
                "ề": ["e","ê"],
                "ể": ["e","ê"],
                "ễ": ["e","ê"],
                "ệ": ["e","ê"],
                "í": ["i"],
                "ì": ["i"],
                "ỉ": ["i"],
                "ĩ": ["i"],
                "ị": ["i"],
                "ó": ["o"],
                "ò": ["o"],
                "ỏ": ["o"],
                "õ": ["o"],
                "ọ": ["o"],
                "ô": ["o"],
                "ố": ["o","ô"],
                "ồ": ["o","ô"],
                "ổ": ["o","ô"],
                "ỗ": ["o","ô"],
                "ộ": ["o","ô"],
                "ú": ["u"],
                "ù": ["u"],
                "ủ": ["u"],
                "ũ": ["u"],
                "ụ": ["u"],
                "ư": ["u"],
                "ứ": ["u","ư"],
                "ừ": ["u","ư"],
                "ử": ["u","ư"],
                "ữ": ["u","ư"],
                "ự": ["u","ư"],
                "ý": ["y"],
                "ỳ": ["y"],
                "ỷ": ["y"],
                "ỹ": ["y"],
                "ỵ": ["y"],
                "đ": ["d"]
            }
        }
        if lang in all_rules.keys():
            self.rules=all_rules[lang]
        else:
            raise ValueError("Language: "+self.params["language"]+" not found in all_rules. keys present are:",all_rules.keys())
        
        self.tokenizer=tokenizer

    def deaccent(self, seq, goal_str, keep_rate=0.2, random_rate=None):
        if random_rate!=None:
            p=np.random.geometric(random_rate)/10
            while p>1:
                p=np.random.geometric(random_rate)/10
            keep_rate=p


        for i,ch in enumerate(list(goal_str)):
            if ch in self.rules.keys():
                if random.random()>keep_rate:
                    #choose which character in rule list to change to (-1 means no cahnge)
                    idx=-1
                    while keep_rate<random.random() and idx<len(self.rules[ch])-1:
                        idx+=1
                    #change to that caharacter
                    if idx>=0:
                        seq[i]=self.tokenizer.encode(self.rules[ch][idx])[0]
        
        return seq