python render.py  --dataset_name render  --root_dir photo/semantic/$1.png    --N_importance 64 --img_wh 256 256   --model photongp_semantic_spade_background_1point_ndc_tri_wocnn_anti --N_sample 64 --chunk 6553600  --ckpt_path ckpts/scene_$1/last.ckpt --scene_name pytorch3d_ngp_semantic_stage2_auto_gan_1p_ndc_tri_wocnn_$1  --batch_size 4 --sky_th $3 --demo --test_name $2