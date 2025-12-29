# Task-dependent cortical oscillatory dynamics in functional constipation
Functional constipation (FC) is a common functional gastrointestinal disorder thought to arise from the brain-gut axis dysfunction, yet direct human neurophysiological evidence is lacking. We recorded high-density electroencephalography (EEG) data in 21 FC patients and 37 healthy controls across resting, cognitive, and defecation-related tasks. We observed that FC patients displayed a consistent, task-dependent signature compared with healthy controls. At the regional level, FC patients exhibited increased alpha during both resting and defecation-related tasks, reduced temporal gamma during defecation-related tasks, as well as elevated temporal theta during the cognitive task. At the global level, we found altered network properties, such as global efficiency in the delta and beta band networks during resting and defecation-related tasks. These findings establish a direct neurophysiological link between specific, condition-dependent perturbations in cortical rhythm activity and FC pathophysiology. Our work implicates the brain-gut axis in symptom generation and opens a path toward EEG-based biomarkers and targeted neuromodulatory therapies.

## Installation
1. Create an environment with Python version 3.10 or higher
```bash
conda create -n env_name python=3.10
```
2. Install pip package
```bash
pip install -r requirements.txt
```

## Usage
The application code for EEG data analysis:
1. Python file: 
draw_topmap.py
-  The script extracts relative power features of different task events in various EEG frequency bands (Delta, Theta, etc.), and then plot the corresponding topographical maps of the brain.
```bash
python draw_topmap.py
```
2. Python file: 
draw_features_relativepower.py
-  The script reads the extracted relative power and performs inter-group statistical analyses on the frequency-band-specific relative power of each brain region under each task, then plots bar charts with significance annotations.
```bash
python draw_features_relativepower.py
```
3. Python file: 
draw_features_network.py
-  The script reads the extracted topological features and performs inter-group statistical analyses on the frequency-band-specific topological features under each task, then plots box charts with significance annotations.
```bash
python draw_features_network.py
```
