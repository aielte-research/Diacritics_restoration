for f in ./*.json ; do
python clean_sentences.py -f "$f" &
done

wait