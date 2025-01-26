import json
import matplotlib.pyplot as plt

with open('defense_model_results.json') as f:
    defense_data = json.load(f)

with open('attack_results.json') as f:
    attack_data = json.load(f)

categories = ['successful_attacks', 'failed_attacks', 'skipped_attacks', 'original_accuracy', 'attack_success_rate']
defense_values = [defense_data[cat] for cat in categories]
attack_values = [attack_data[cat] for cat in categories]

x = range(len(categories))
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x, defense_values, width, label='Defense Model')
bars2 = ax.bar([p + width for p in x], attack_values, width, label='Attack Model')

ax.set_xlabel('Categories')
ax.set_ylabel('Values')
ax.set_title('Comparison of Defense and Attack Model Results')
ax.set_xticks([p + width / 2 for p in x])
ax.set_xticklabels(categories)
ax.legend()

plt.show()
