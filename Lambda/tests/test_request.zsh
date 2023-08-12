epoch=1
worker_number=1

for i in {0..7}; do
    python3 app.py --worker "${i}/${worker_number}" --epoch ${epoch} &
done

wait

curl -XGET http://127.0.0.1:8080/check
