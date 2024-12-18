# 1、Encode reference audio:
src_audio="D:\PythonProject\vo_hutao_draw_appear.wav"

python tools/vqgan/inference.py \
    -i {src_audio} \
    --checkpoint-path "checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
# generate fake.npy: prompt codes \
#          fake.wav: prompt wav

# 2、Generate semantic tokens from text:
python tools/llama/generate.py \
    --text "hello world" \
    --prompt-text "The text corresponding to reference audio" \
    --prompt-tokens "fake.npy" \
    --checkpoint-path "checkpoints/fish-speech-1.4" \
    --num-samples 2
    # --compile
# llama_style gpt:predict codes

# 3、Generate speech from semantic tokens:

python tools/vqgan/inference.py \
    -i "codes_0.npy" \
    --checkpoint-path "checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"

# vqgan decoder: input codes
#                output wav