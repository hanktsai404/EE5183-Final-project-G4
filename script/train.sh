# --- NNStockTrainer Args ---
seed=7
device_id="0"
log_path="./log/"
lr=0.001
epoch=100
checkpoint_path="./checkpoint/"

# --- NNStockSolver Args ---
method="Prophet" # [Algorithm]: Please see main.py for more details
feat_num=24
time_step=40 # [TimeStep]: How many days as data to predict MA
moving_average=7 # [MovingAverage]: How many days to average
stat_dir='./data_stats/'
train_percent=1.0
data_dir='./firms/'
batch_size=32
num_workers=0
# --pin_memory
# --shuffle

# --- Function Mode ---
# --train
# --plot_pred # Replace --train with this flag for inference-only

# --- Additional Args ---
firm_table='./data/Market_Value_Table.csv'
firm_num=20 # [Firm]: How many companies to predict (max=20). Set to 1 for fixing code.

# --- Make Directory ---
mkdir -p $log_path
mkdir -p $checkpoint_path
mkdir -p $stat_dir

python main.py \
	--seed $seed \
	--device_id $device_id \
	--log_path $log_path \
	--lr $lr \
	--epoch $epoch \
	--checkpoint_path $checkpoint_path \
	--method $method \
	--feat_num $feat_num \
	--time_step $time_step \
	--moving_average $moving_average \
	--stat_dir $stat_dir \
	--train_percent $train_percent \
	--data_dir $data_dir \
	--batch_size $batch_size \
	--num_workers $num_workers \
	--firm_table $firm_table \
	--firm_num $firm_num \
	--pin_memory \
	--shuffle \
	--train 