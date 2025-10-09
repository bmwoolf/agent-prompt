# agent-prompt
a recyclable agent template for building models in Python


## how to execute 
add the `agent_prompt.md` to Cursor's context, then customize and paste this into the message window:

```
use the full prompt in ⟨PATH_TO_AGENT_PROMPT_MD⟩ as system instructions.
do not summarize or reinterpret.

environment profile is produced by ⟨PATH_TO_CONFIG_SCRIPT⟩ → writes ⟨PATH_TO_CONFIG_YML⟩.
if ⟨PATH_TO_CONFIG_YML⟩ is missing, run:
chmod +x ⟨PATH_TO_CONFIG_SCRIPT⟩ && ⟨PATH_TO_CONFIG_SCRIPT⟩

dataset profile is in ⟨PATH_TO_DATASET_MD⟩ (schema, preprocessing, splits, subset rules).

project:
- name: ⟨PROJECT_NAME⟩
- objective: ⟨ONE_LINE_OBJECTIVE⟩
- scope overrides (optional): ⟨KEY_OVERRIDES_OR_EMPTY⟩

execute now to scaffold the project per the conventions in the system prompt.
merge environment defaults from ⟨PATH_TO_CONFIG_YML⟩ and dataset metadata from ⟨PATH_TO_DATASET_MD⟩ when generating code and configs.
follow the request/response contract exactly.
```

#### example
```
use the full prompt in config/agent_prompt.md as system instructions.
do not summarize or reinterpret.

environment profile is produced by config/config.sh → writes config.yml.
if config.yml is missing, run:
chmod +x config/config.sh && config/config.sh

dataset profile is in config/dataset.md.

project:
- name: "vae_scRNAseq"
- objective: train a torch VAE on PBMC10x, use latent embeddings for cell annotation, and evaluate vs known labels.
- scope overrides: none

execute now to scaffold the project per the conventions in agent_prompt.md.
merge environment defaults from config.yml and dataset metadata from config/dataset.md when generating code and configs.
follow the request/response contract exactly.
```