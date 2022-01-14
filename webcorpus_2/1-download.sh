mkdir wiki
cd wiki
for NUM in {1..168..1}
do
wget "https://nessie.ilab.sztaki.hu/~ndavid/Webcorpus2/wiki_$(printf "%04d" $NUM).tsv.gz"
done
cd ..

mkdir 2019
cd 2019
for NUM in {1..600..1}
do
wget "https://nessie.ilab.sztaki.hu/~ndavid/Webcorpus2/2019_$(printf "%04d" $NUM).tsv.gz"
done
cd ..

mkdir 2017_2018
cd 2017_2018
for NUM in {1..3697..1}
do
wget "https://nessie.ilab.sztaki.hu/~ndavid/Webcorpus2/2017_2018_$(printf "%04d" $NUM).tsv.gz"
done
cd ..