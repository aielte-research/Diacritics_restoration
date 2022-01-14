for d in ./wiki*/ ; do
cd "$d"
for f in ./*.json ; do
python ../clean_sentences.py -f "$f" &
done
cd ..
done

for d in ./2019*/ ; do
cd "$d"
for f in ./*.json ; do
python ../clean_sentences.py -f "$f" &
done
cd ..
done

for d in ./2017*/ ; do
cd "$d"
for f in ./*.json ; do
python ../clean_sentences.py -f "$f" &
done
cd ..
done

wait