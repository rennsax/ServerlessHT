# check tmux installed
# if ! command -v tmux > /dev/null; then
#     echo "no tmux"
#     exit 1
# fi

# tmux new-session -d -s "tuning-server" 'python3 -m flask -A server run --debug --port 8080 --host 0.0.0.0'

# tmux new-window 'python3 main.py 2>&1 | tee -i worker.log'

if [[ "$#" -ne 9 ]]; then
    echo "Not enough arguments!"
    exit 1
fi

if ! ls EC2 Lambda > /dev/null 2>&1; then
    echo "The directory is incorrect!"
    exit 1
fi

export EC2_PROXY_USE_CLI=1

index=$6

args="\
--worker-number $1 \
--function-name $2 \
--data-size $3 \
--epoch $4 \
--port $5 \
--output-file output/result_${index}.txt \
--batch-size $7 \
--momentum $8 \
--learning-rate $9 \
"

eval "python3 EC2/server.py $args < /dev/null >| EC2/log/server_${index}.log 2>&1 &|"
server_pid=$!

eval "python3 EC2/main.py $args 2>&1 | tee -i EC2/log/worker_${index}.log &"
wait $!
kill -9 $server_pid
