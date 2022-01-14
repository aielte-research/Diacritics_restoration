for NUM in {0..9..1}
do
echo extracting web2-4p-$NUM.tar.gz
mkdir web2-4p-$NUM
tar -xzkf "web2-4p-$NUM.tar.gz" -C "./web2-4p-$NUM" --checkpoint=.1000 &
done
wait

for NUM in {0..9..1}
do
rm web2-4p-$NUM.tar.gz
echo web2-4p-$NUM.tar.gz removed
done