import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

test_acc = [0.9005101919174194,0.944727897644043,0.9549319744110107,0.9098639488220215,
            0.930272102355957,0.9404761791229248, 0.8934859037399292, 0.956250011920929]
model = ['Small x1','Small x1.25','Small x1.50','Small x1.75','Small x2','Small x2.25', 'Medium', 'Large']
# Set the figure size
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.size'] = 12
plt.figure(figsize=(15, 8))
plt.scatter(model, test_acc, color='blue', marker='o', label='Test Accuracy')
plt.plot(model, test_acc, color='blue', linestyle='-', linewidth=1)

plt.xlabel('Model')
plt.ylabel('Test Accuracy')
# plt.title('Test Accuracies for Different Models')
plt.ylim(0.85, 1.0)  # Adjust the y-axis limits as needed

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig('small_patch_model_test_acc.png',dpi=400)
plt.close()


directory_path = os.getcwd()
# Get a list of all files in the directory
files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

# Filter files with 'test_' in their names
test_files = [f for f in files if 'test_' in f]

# Define a regular expression pattern to extract the float value
accuracy_pattern = r'Test Accuracy: (\d+\.\d+)'

# Define a regular expression pattern to extract 'small_2-25'
name_pattern = r'test_results_(small_\d+-\d+)\.txt'

# Iterate through the test files and extract the float value and name
for file_name in sorted(test_files):
    file_path = os.path.join(directory_path, file_name)
    
    with open(file_path, 'r') as file:
        content = file.read()
        
        # Search for the accuracy pattern in the file content
        accuracy_match = re.search(accuracy_pattern, content)

        
        if accuracy_match:
            # Extract the float value from the accuracy pattern
            test_accuracy = float(accuracy_match.group(1))
            
            # Print or use the results as needed
            print(f'Test Accuracy in {file_name}: {test_accuracy}')
        else:
            print(f'No Test Accuracy or Name found in {file_name}')