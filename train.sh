# Human3.6M
python train_net.py --cfg_file configs/aninerf_s1p.yaml exp_name aninerf_s1p resume False
python train_net.py --cfg_file configs/aninerf_s1p.yaml exp_name aninerf_s1p_full resume False aninerf_animation True init_aninerf aninerf_s1p

python train_net.py --cfg_file configs/aninerf_s5p.yaml exp_name aninerf_s5p resume False
python train_net.py --cfg_file configs/aninerf_s5p.yaml exp_name aninerf_s5p_full resume False aninerf_animation True init_aninerf aninerf_s5p

python train_net.py --cfg_file configs/aninerf_s6p.yaml exp_name aninerf_s6p resume False
python train_net.py --cfg_file configs/aninerf_s6p.yaml exp_name aninerf_s6p_full resume False aninerf_animation True init_aninerf aninerf_s6p

python train_net.py --cfg_file configs/aninerf_s7p.yaml exp_name aninerf_s7p resume False
python train_net.py --cfg_file configs/aninerf_s7p.yaml exp_name aninerf_s7p_full resume False aninerf_animation True init_aninerf aninerf_s7p

python train_net.py --cfg_file configs/aninerf_s8p.yaml exp_name aninerf_s8p resume False
python train_net.py --cfg_file configs/aninerf_s8p.yaml exp_name aninerf_s8p_full resume False aninerf_animation True init_aninerf aninerf_s8p

python train_net.py --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p resume False
python train_net.py --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p_full resume False aninerf_animation True init_aninerf aninerf_s9p

python train_net.py --cfg_file configs/aninerf_s11p.yaml exp_name aninerf_s11p resume False
python train_net.py --cfg_file configs/aninerf_s11p.yaml exp_name aninerf_s11p_full resume False aninerf_animation True init_aninerf aninerf_s11p

# ZJU-MoCap
python train_net.py --cfg_file configs/aninerf_313.yaml exp_name aninerf_313 resume False
python train_net.py --cfg_file configs/aninerf_313.yaml exp_name aninerf_313_full resume False aninerf_animation True init_aninerf aninerf_313

python train_net.py --cfg_file configs/aninerf_315.yaml exp_name aninerf_315 resume False
python train_net.py --cfg_file configs/aninerf_315.yaml exp_name aninerf_315_full resume False aninerf_animation True init_aninerf aninerf_315

python train_net.py --cfg_file configs/aninerf_377.yaml exp_name aninerf_377 resume False
python train_net.py --cfg_file configs/aninerf_377.yaml exp_name aninerf_377_full resume False aninerf_animation True init_aninerf aninerf_377

python train_net.py --cfg_file configs/aninerf_386.yaml exp_name aninerf_386 resume False
python train_net.py --cfg_file configs/aninerf_386.yaml exp_name aninerf_386_full resume False aninerf_animation True init_aninerf aninerf_386


# 
python train_net.py --cfg_file configs/aninerf_315.yaml exp_name aninerf_315 resume False gpus "9,"
python train_net.py --cfg_file configs/aninerf_315.yaml exp_name aninerf_315_full resume False aninerf_animation True init_aninerf aninerf_315 gpus "9,"
python train_net.py --cfg_file configs/aninerf_315.yaml exp_name aninerf_315_full_ind resume False aninerf_animation_ind True init_aninerf aninerf_315 gpus "9,"

python run.py --type evaluate --cfg_file configs/aninerf_315.yaml exp_name aninerf_315 resume True gpus "9,"
python run.py --type evaluate --cfg_file configs/aninerf_315.yaml exp_name aninerf_315_full resume True aninerf_animation_ind True init_aninerf aninerf_315 gpus "9," test.frame_sampler_interval 20 test_novel_pose True


for ID in 386; do python train_net.py --cfg_file configs/aninerf_$ID.yaml exp_name aninerf_$ID resume True gpus "0,"; done;

for ID in 313 315 377 386; do python run.py --type evaluate --cfg_file configs/aninerf_$ID.yaml exp_name aninerf_$ID resume True gpus "0," eval_whole_img True; done;

python train_net.py --cfg_file configs/aninerf_313.yaml exp_name aninerf_313_ood resume False aninerf_animation True init_aninerf aninerf_313 gpus "1," train.epoch 400
python train_net.py --cfg_file configs/aninerf_315.yaml exp_name aninerf_315_ood resume False aninerf_animation True init_aninerf aninerf_315 gpus "2," train.epoch 400
python train_net.py --cfg_file configs/aninerf_377.yaml exp_name aninerf_377_ood resume False aninerf_animation True init_aninerf aninerf_377 gpus "3," train.epoch 400
python train_net.py --cfg_file configs/aninerf_386.yaml exp_name aninerf_386_ood resume False aninerf_animation True init_aninerf aninerf_386 gpus "4," train.epoch 400

python train_net.py --cfg_file configs/aninerf_313.yaml exp_name aninerf_313_ind resume False aninerf_animation_ind True init_aninerf aninerf_313 gpus "5," train.epoch 400
python train_net.py --cfg_file configs/aninerf_315.yaml exp_name aninerf_315_ind resume False aninerf_animation_ind True init_aninerf aninerf_315 gpus "6," train.epoch 400
python train_net.py --cfg_file configs/aninerf_377.yaml exp_name aninerf_377_ind resume False aninerf_animation_ind True init_aninerf aninerf_377 gpus "7," train.epoch 400
python train_net.py --cfg_file configs/aninerf_386.yaml exp_name aninerf_386_ind resume False aninerf_animation_ind True init_aninerf aninerf_386 gpus "8," train.epoch 400


python run.py --type evaluate --cfg_file configs/aninerf_313.yaml exp_name aninerf_313_ood resume True aninerf_animation True init_aninerf aninerf_313 gpus "1," eval_whole_img True test_novel_pose True
python run.py --type evaluate --cfg_file configs/aninerf_315.yaml exp_name aninerf_315_ood resume True aninerf_animation True init_aninerf aninerf_315 gpus "2," eval_whole_img True test_novel_pose True
python run.py --type evaluate --cfg_file configs/aninerf_377.yaml exp_name aninerf_377_ood resume True aninerf_animation True init_aninerf aninerf_377 gpus "3," eval_whole_img True test_novel_pose True
python run.py --type evaluate --cfg_file configs/aninerf_386.yaml exp_name aninerf_386_ood resume True aninerf_animation True init_aninerf aninerf_386 gpus "4," eval_whole_img True test_novel_pose True

python run.py --type evaluate --cfg_file configs/aninerf_313.yaml exp_name aninerf_313_ind resume True aninerf_animation_ind True init_aninerf aninerf_313 gpus "5," eval_whole_img True test_novel_ind_pose True
python run.py --type evaluate --cfg_file configs/aninerf_315.yaml exp_name aninerf_315_ind resume True aninerf_animation_ind True init_aninerf aninerf_315 gpus "6," eval_whole_img True test_novel_ind_pose True
python run.py --type evaluate --cfg_file configs/aninerf_377.yaml exp_name aninerf_377_ind resume True aninerf_animation_ind True init_aninerf aninerf_377 gpus "7," eval_whole_img True test_novel_ind_pose True
python run.py --type evaluate --cfg_file configs/aninerf_386.yaml exp_name aninerf_386_ind resume True aninerf_animation_ind True init_aninerf aninerf_386 gpus "8," eval_whole_img True test_novel_ind_pose True
