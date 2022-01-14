for NUM in {0..9..1}
do
echo editing files in web2-4p-$NUM &
python edit_files.py -f web2-4p-$NUM &
done
wait
