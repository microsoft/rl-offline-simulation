# Run notebook to generate data

# Train HOMER
# cd into this directory first
# python train_homer_encoder.py \
#     --num_epochs=20 \
#     --seed=0 \
#     --batch_size=64 \
#     --latent_size=8 \
#     --hidden_size=64 \
#     --lr=1e-3 \
#     --weight_decay=0.0 \
#     --temperature_decay=False \
#     --output_dir='outputs/models' \
#     --num_samples=1000

python train_homer_encoder.py \
    --num_epochs=1000 \
    --seed=0 \
    --batch_size=64 \
    --latent_size=16 \
    --hidden_size=64 \
    --lr=1e-3 \
    --weight_decay=0.0 \
    --temperature_decay=False \
    --output_dir='outputs/models'
    
python train_homer_encoder.py \
    --num_epochs=1000 \
    --seed=0 \
    --batch_size=64 \
    --latent_size=32 \
    --hidden_size=64 \
    --lr=1e-3 \
    --weight_decay=0.0 \
    --temperature_decay=False \
    --output_dir='outputs/models'