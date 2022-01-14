THREADS=16

cd wiki
for ((NUM_=1; NUM_<=168; NUM_=NUM_+THREADS))
do
	for ((NUM=NUM_; NUM<NUM_+THREADS; NUM++))
	do
		echo extracting "wiki_$(printf "%04d" $NUM_).tsv.gz"
		gzip -d "wiki_$(printf "%04d" $NUM).tsv.gz" -f &
	done
	wait
	for ((NUM=NUM_; NUM<NUM_+THREADS; NUM++))
	do
		python ../get_text.py -f wiki_$(printf "%04d" $NUM).tsv &
	done
	wait
	for ((NUM=NUM_; NUM<NUM_+THREADS; NUM++))
	do
		rm wiki_$(printf "%04d" $NUM).tsv &
	done
	
done
cd ..

cd 2019
for ((NUM_=1; NUM_<=600; NUM_=NUM_+THREADS))
do
	for ((NUM=NUM_; NUM<NUM_+THREADS; NUM++))
	do
		echo extracting "2019_$(printf "%04d" $NUM).tsv.gz"
		gzip -d "2019_$(printf "%04d" $NUM).tsv.gz" -f &
	done
	wait
	for ((NUM=NUM_; NUM<NUM_+THREADS; NUM++))
	do
		python ../get_text.py -f 2019_$(printf "%04d" $NUM).tsv &
	done
	wait
	for ((NUM=NUM_; NUM<NUM_+THREADS; NUM++))
	do
		rm 2019_$(printf "%04d" $NUM).tsv &
	done
	
done
cd ..

cd 2017_2018
for ((NUM_=1; NUM_<=3697; NUM_=NUM_+THREADS))
do
	for ((NUM=NUM_; NUM<NUM_+THREADS; NUM++))
	do
		echo extracting "2017_2018_$(printf "%04d" $NUM).tsv.gz"
		gzip -d "2017_2018_$(printf "%04d" $NUM).tsv.gz" -f &
	done
	wait
	for ((NUM=NUM_; NUM<NUM_+THREADS; NUM++))
	do
		python ../get_text.py -f 2017_2018_$(printf "%04d" $NUM).tsv &
	done
	wait
	for ((NUM=NUM_; NUM<NUM_+THREADS; NUM++))
	do
		rm 2017_2018_$(printf "%04d" $NUM).tsv &
	done
	
done
cd ..
