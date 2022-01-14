To generate the data run the bash files in order, e.g.:
```
bash 1-download.sh
```
Edit `max_len=500` in `get_text.py` to change per sequence length limit.

`5-clean_sequences.sh` is optional, it throws out sequences wich may contain sentences without proper diacritics.

`6-create_cuts_sentences.sh` is also optional.