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


# 2.merge
task_vector_sum = sum(task_vector_list)
scaling_coef = 0.2
merged_model = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)


save_dir = f'model_merging/fft-merge'
os.makedirs(save_dir, exist_ok=True)
save_path = f'{save_dir}/asr_fft-merge_{num_language}lang-{hours}hour_scaling{scaling_coef}.pt'
torch.save(merged_model, save_path)
print(f'save {save_path}')