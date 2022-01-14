echo concatenating files in wiki
python concatenate.py -f wiki
rm -r wiki
echo wiki removed

echo concatenating files in 2019
python concatenate.py -f 2019
rm -r 2019
echo 2019 removed

echo concatenating files in 2017_2018
python concatenate.py -f 2017_2018
rm -r 2017_2018
echo 2017_2018 removed
