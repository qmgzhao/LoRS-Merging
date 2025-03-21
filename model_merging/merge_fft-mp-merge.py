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


# 2.magnitude pruning
def magnitude_prune(weight, ratio):
    k = int(weight.numel() * ratio)
    if k > 0:
        _, topk_indices = weight.abs().view(-1).topk(k)
        mask = torch.zeros_like(weight.view(-1), dtype=torch.bool)
        mask[topk_indices] = True
        return torch.where(mask, weight.view(-1), torch.tensor(0.0, device=weight.device)).view_as(weight)
    return torch.zeros_like(weight)


ratio = 0.2
for task_vector in task_vector_list:
    for name, param in list(task_vector.vector.items()):
        if ('attn.' in name or 'mlp.' in name) and 'weight' in name:
            delta_W = task_vector.vector[name]
            task_vector.vector[name] = magnitude_prune(delta_W, ratio)


# 3.merge
task_vector_sum = sum(task_vector_list)
scaling_coef = 0.2
merged_model = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)


save_dir = 'model_merging/fft-mp-merge'
os.makedirs(save_dir, exist_ok=True)
save_path = f'{save_dir}/asr_fft-mp-merge_{num_language}lang-{hours}hour_scaling{scaling_coef}_ratio{ratio}.pt'
torch.save(merged_model, save_path)
print(f'save {save_path}')