import pandas as pd
import matplotlib.pyplot as plt

# Add correct file path
file = pd.read_csv(f'./model_output/2024-04-23_weight_E100_200_0.000633_loss.csv')

lines = file.plot.line(x='Epoch', y=['Train Loss', 'Validation Loss'])
plt.title('Model learning curves')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()