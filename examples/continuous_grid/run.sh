# Generate data
python examples/continuous_grid/random_agent_rollout.py

# Train HOMER
python examples/continuous_grid/train_homer_encoder.py \
    --num_samples=50000 `# for debugging` \
    --num_epochs=100 \
    --seed=0 \
    --batch_size=64 \
    --latent_size=25 \
    --hidden_size=64 \
    --lr=1e-3 \
    --weight_decay=0.0 \
    --temperature_decay=True \
    --output_dir='outputs/models'

python examples/continuous_grid/train_homer_encoder.py \
    --num_epochs=1000 \
    --seed=0 \
    --batch_size=64 \
    --latent_size=50 \
    --hidden_size=64 \
    --lr=1e-3 \
    --weight_decay=0.0 \
    --temperature_decay=False \
    --output_dir='outputs/models' \
    --num_samples=100000