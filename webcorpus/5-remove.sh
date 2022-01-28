for NUM in {0..9..1}
do
echo removing web2-4p-$NUM &
rm -r web2-4p-$NUM &
done
wait
echo done