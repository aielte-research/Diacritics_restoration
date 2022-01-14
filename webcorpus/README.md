To generate the data run the bash files in order, e.g.:
```
bash 1-download.sh
```
Edit `max_chr_count = 500` in `concatenate.py` to change per sequence length limit.

`6-clean_sentences.sh` is optional, it throws out sequences wich may contain sentences without proper diacritics.
