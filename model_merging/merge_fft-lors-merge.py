import os
import torch
from task_vectors import TaskVector


pretrained_checkpoint = f'/HOME/scw7196/run/projects/whisper/models/small/fp32.small.cuda.pth'
finetuned_dir = f'/HOME/scw7196/run/projects/whisper/model_merging/asr'


# language_list = ["id", "nl", "pt", "ru", "sv-SE"]  # 5-language
language_list = ["ca", "de", "es", "fr", "it"]  # 5-language
language_string = "-".join(language_list)

num_language = len(language_list)


# 1.compute task vector
hours = 5
task_vector_list = []
for language in language_list:
    finetuned_checkpoint = os.path.join(finetuned_dir, f'{hours}hour_{language}', 'model.acc.best')
    print(f'finetuned_checkpoint: {finetuned_checkpoint}')
    task_vector = TaskVector(pretrained_checkpoint, finetuned_checkpoint)
    task_vector_list.append(task_vector)


# 2.LoRS pruning
def magnitude_prune(weight, ratio):
    k = int(weight.numel() * ratio)
    if k > 0:
        _, topk_indices = weight.abs().view(-1).topk(k)
        mask = torch.zeros_like(weight.view(-1), dtype=torch.bool)
        mask[topk_indices] = True
        return torch.where(mask, weight.view(-1), torch.tensor(0.0, device=weight.device)).view_as(weight)
    return torch.zeros_like(weight)


ratio = 0.2  # magnitude pruning
index = 7  # singular value pruning

for task_vector in task_vector_list:
    for name, param in list(task_vector.vector.items()):
        if ('attn.' in name or 'mlp.' in name) and 'weight' in name:
            delta_W = task_vector.vector[name]

            # 2.1 SVP
            weight_u, weight_sigma, weight_vt = torch.linalg.svd(delta_W, full_matrices=True)

            weight_sigma_top = weight_sigma[:index]
            weight_u_top = weight_u[:, :index]
            weight_vt_top = weight_vt[:index, :]

            L = weight_u_top @ torch.diag(weight_sigma_top) @ weight_vt_top

            # 2.2 residual matrix
            S = delta_W - L

            # 2.3 MP
            S = magnitude_prune(S, ratio)

            # 2.4 add
            task_vector.vector[name] = L + S


# 3.merge
task_vector_sum = sum(task_vector_list)
scaling_coef = 0.2
merged_model = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)


save_dir = 'model_merging/fft-lors-merge'
os.makedirs(save_dir, exist_ok=True)
save_path = f'{save_dir}/asr_fft-lors-merge_{num_language}lang-{hours}hour_scaling{scaling_coef}_svp-index{index}_mp-ratio{ratio}.pt'
torch.save(merged_model, save_path)
print(f'save {save_path}')