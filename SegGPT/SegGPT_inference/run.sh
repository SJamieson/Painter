mkdir -p $4
PROMPTS=$(ls $2/* -qb)
TARGETS=$(ls $3/* -qb)
echo $PROMPTS
echo $TARGETS
python seggpt_inference.py --input_video $1 --prompt_image $PROMPTS --prompt_target $TARGETS --output_dir $4
