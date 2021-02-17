data_path=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/TianTian/data/birds
save_root=outputs/attention_layer4/birds/full
cfg_file=Confg/birds_train_batch.yml
result_file=full.text
seed=200
lr=0.0001
wd=1e-5
workers=8
batch_size=64
n_heads=1
n_epochs=120
start_epoch=0
lr_decay=50
smooth_gamm3=13.0
gamma_clss=0.5


python3 run.py --data_path $data_path \
              --save_root $save_root \
      			  --cfg_file $cfg_file\
      			  --lr $lr\
      			  --manualSeed $seed\
      			  --weight-decay $wd\
              --WORKERS $workers\
              --batch_size $batch_size\
			  --n_heads $n_heads\
			  --n_epochs $n_epochs\
			  --result_file $result_file\
			  --start_epoch $start_epoch\
        --lr_decay $lr_decay \
		--smooth_gamm3 $smooth_gamm3
                        
                               
