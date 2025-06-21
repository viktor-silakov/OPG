#!/usr/bin/env python3

with open('finetune_tts.py', 'r') as f:
    lines = f.readlines()

# Найти строку с resume_checkpoint и добавить новые параметры
for i, line in enumerate(lines):
    if 'resume_checkpoint,' in line and i > 760:
        lines.insert(i+1, '            getattr(args, "early_stopping_patience", None),\n')
        lines.insert(i+2, '            getattr(args, "save_every_n_steps", 100),\n')
        lines.insert(i+3, '            getattr(args, "log_every_n_steps", 10)\n')
        break

with open('finetune_tts.py', 'w') as f:
    f.writelines(lines)

print("Parameters added successfully!") 