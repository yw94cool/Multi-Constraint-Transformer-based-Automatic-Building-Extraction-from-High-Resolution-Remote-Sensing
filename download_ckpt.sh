wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz

mkdir -p ./model/vit_checkpoint/imagenet21k

mv R50+ViT-B_16.npz ./model/vit_checkpoint/imagenet21k/
