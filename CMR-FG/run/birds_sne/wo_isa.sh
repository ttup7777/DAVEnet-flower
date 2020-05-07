data_path=/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/birds
save_root=outputs/01_Baseline/birds/wo_isa
cfg_file=Confg/bird_sne.yml
result_file=wo_isa.text
seed=200
lr=0.0001
wd=1e-4
workers=8
batch_size=128
n_heads=1
n_epochs=80
start_epoch=40
lr_decay=50
smooth_gamm3=10.0


python run.py --data_path $data_path \
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
                        
                               
