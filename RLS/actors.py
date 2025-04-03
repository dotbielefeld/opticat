import torch.nn as nn
import torch.nn.functional as F

class GLUBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(GLUBlock, self).__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, output_dim*2)  # GLU halves the dimension, prepare for it
        self.linear2 = nn.Linear(output_dim, output_dim)  # Maintain dimension after GLU
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x
        out = self.norm(x)
        out = F.glu(self.linear1(out), dim=1)
        out = self.dropout(self.linear2(out))
        out += identity  # Adding the input to the output (Residual Connection)
        return out


class Actor_GLU_BETA(nn.Module):
    def __init__(self, state_dim, num_categories, num_continuous_actions, num_blocks, num_neurons, dropout_rate=0):
        super(Actor_GLU_BETA, self).__init__()
        self.num_blocks = num_blocks
        self.num_neurons = num_neurons

        # Initial Linear + GLU layers
        self.initial = nn.Sequential(
            nn.Linear(state_dim, self.num_neurons * 2),
            nn.GLU(dim=1),
            nn.Linear(self.num_neurons, self.num_neurons * 2),
            nn.GLU(dim=1),
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            GLUBlock(self.num_neurons, self.num_neurons, dropout_rate=dropout_rate) for _ in range(self.num_blocks)
        ])

        # Final normalization and transformation
        self.final_norm = nn.LayerNorm(self.num_neurons)
        self.final_linear = nn.Linear(self.num_neurons, self.num_neurons*2) #+*2 for glu
       # self.final_norm_2 = nn.LayerNorm(self.num_neurons)

        # Output heads
        self.categorical_heads = nn.ModuleList([nn.Linear(self.num_neurons, n) for n in num_categories])
        self.continuous_heads_alpha = nn.ModuleList([nn.Linear(self.num_neurons, 1) for _ in range(num_continuous_actions)])
        self.continuous_heads_beta = nn.ModuleList([nn.Linear(self.num_neurons, 1) for _ in range(num_continuous_actions)])


    def forward(self, state):
        x = self.initial(state)
        for block in self.blocks:
            x = block(x)

        x = F.glu(self.final_linear(self.final_norm(x)), dim=1)

        cat_logits = [head(x) for head in self.categorical_heads] #No activation here since I do it when sampling

        continuous_outputs_alpha = [F.softplus(head(x)) for head in self.continuous_heads_alpha]
        continuous_outputs_beta = [F.softplus(head(x)) for head in self.continuous_heads_beta]

        return cat_logits, continuous_outputs_alpha, continuous_outputs_beta


class Actor_GLU_NORMAL(nn.Module):
    def __init__(self, state_dim, num_categories, num_continuous_actions, num_blocks, num_neurons, dropout_rate=0):
        super(Actor_GLU_NORMAL, self).__init__()
        self.num_blocks = num_blocks
        self.num_neurons = num_neurons

        # Initial Linear + GLU layers
        self.initial = nn.Sequential(
            nn.Linear(state_dim, self.num_neurons * 2),
            nn.GLU(dim=1),
            nn.Linear(self.num_neurons, self.num_neurons * 2),
            nn.GLU(dim=1),
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            GLUBlock(self.num_neurons, self.num_neurons, dropout_rate=dropout_rate) for _ in range(self.num_blocks)
        ])

        # Final normalization and transformation
        self.final_norm = nn.LayerNorm(self.num_neurons)
        self.final_linear = nn.Linear(self.num_neurons, self.num_neurons * 2)  # *2 for GLU

        # Output heads
        self.categorical_heads = nn.ModuleList([nn.Linear(self.num_neurons, n) for n in num_categories])
        # Replace alpha and beta with mean and std heads
        self.continuous_heads_mean = nn.ModuleList([nn.Linear(self.num_neurons, 1) for _ in range(num_continuous_actions)])
        self.continuous_heads_std = nn.ModuleList([nn.Linear(self.num_neurons, 1) for _ in range(num_continuous_actions)])

    def forward(self, state):
        x = self.initial(state)

        for block in self.blocks:
            x = block(x)

        x = F.glu(self.final_linear(self.final_norm(x)), dim=1)

        # Categorical logits
        cat_logits = [head(x) for head in self.categorical_heads]

        # Continuous actions: Mean and standard deviation
        continuous_means = [head(x) for head in self.continuous_heads_mean]
        continuous_stds = [F.softplus(head(x)) + 1e-6 for head in self.continuous_heads_std]  # Add small epsilon to ensure positivity

        return cat_logits, continuous_means, continuous_stds