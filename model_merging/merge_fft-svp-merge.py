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


# 2.singular value pruning
ratio = 0.01
dim = 768
index = int(dim * ratio)
# index = 1
for task_vector in task_vector_list:
    for name, param in list(task_vector.vector.items()):
        if ('attn.' in name or 'mlp.' in name) and 'weight' in name:
            delta_W = task_vector.vector[name]
            weight_u, weight_sigma, weight_vt = torch.linalg.svd(delta_W, full_matrices=True)

            weight_sigma_top = weight_sigma[:index]
            weight_u_top = weight_u[:, :index]
            weight_vt_top = weight_vt[:index, :]

            delta_W = weight_u_top @ torch.diag(weight_sigma_top) @ weight_vt_top
            task_vector.vector[name] = delta_W


# 3.merge
task_vector_sum = sum(task_vector_list)
scaling_coef = 0.2
merged_model = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)


save_dir = 'model_merging/fft-svp-merge'
os.makedirs(save_dir, exist_ok=True)
save_path = f'{save_dir}/asr_fft-svp-merge_{num_language}lang-{hours}hour_scaling{scaling_coef}_index{index}.pt'
torch.save(merged_model, save_path)
print(f'save {save_path}')