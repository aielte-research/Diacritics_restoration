for NUM in {0..9..1}
do
echo concatenating files in web2-4p-$NUM &
python concatenate.py -f web2-4p-$NUM &
done
wait