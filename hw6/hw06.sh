pip install stylegan2_pytorch pytorch-fid --quiet
stylegan2_pytorch --data ./faces --results_dir ./result --models_dir ./models \
    --network-capacity 64 \
    -attn-layers 1 \
    --num-train-steps 50000

stylegan2_pytorch --generate --num_generate=1000 --image_size=64 --num_image_tiles 1
i=1 && for f in `ls results/default/*-ema.jpg` ; do cp $f $i.jpg; ((i++)); done
tar -zcf /content/images.tgz *.jpg