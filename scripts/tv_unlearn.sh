
ALPHAS=(1.0 1.5 2.0)
BETAS=(1.0)

for ALPHA in "${ALPHAS[@]}"; do
    for BETA in "${BETAS[@]}"; do

        python src/tv_unlearn.py \
        --clean_model_name     Llama-3.2-1B-Instruct \
        --corrupted_model_name Llama-3.2-1B-Instruct \
        --forget_model_name    Llama-3.2-1B-Instruct \
        --retain_model_name    Llama-3.2-1B-Instruct \
        --local_forget_model_path     outputs/Llama-3.2-1B-Instruct_TOFU_QA_forget01 \
        --local_retain_model_path     outputs/Llama-3.2-1B-Instruct_TOFU_QA_retain99 \
        --local_corrupted_model_path outputs/Llama-3.2-1B-Instruct_TOFU_QA_full \
        --alpha ${ALPHA} \
        --beta ${BETA} \
        --save_path outputs/Llama-3.2-1B-Instruct_TOFU_QA_forget01_tv_unlearn_alpha-${ALPHA}_beta-${BETA} \

    done
done