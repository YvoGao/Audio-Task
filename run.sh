for T in 5; do
for dataset in FSC LBS  Ny; do
    for seed in 0 1 2 3 4 5 6 7 8 9 \
            10 11 12 13 14 15 16 17 18 19 \
            20 21 22 23 24 25 26 27 28 29 \
            30 31 32 33 34 35 36 37 38 39 \
            40 41 42 43 44 45 46 47 48 49 \
            50 51 52 53 54 55 56 57 58 59 \
            60 61 62 63 64 65 66 67 68 69 \
            70 71 72 73 74 75 76 77 78 79 \
            80 81 82 83 84 85 86 87 88 89 \
            90 91 92 93 94 95 96 97 98 99 ; do
        query=200
        if [ "$dataset" = "FSC" ]; then
            query=200
        else
            query=100
        fi
        if [ "$dataset" = "LBS" ]; then
            epoch=10
        else
            epoch=3
        fi
        python train.py \
            --dataset $dataset \
            --save_path $dataset"_result/" \
            --model_name or \
            --n_epochs $epoch \
            --query $query \
            --train_batch_size 16 \
            --test_batch_size 200 \
            --exp_name $dataset"_test_session_"$T \
            --seed $seed \
            --session $T \
            --cuda 1 \
            --do_logging
    done
done
done
