''''''''''''''''''''
''''''''''''''''''''
''''''''''''''''''''
''''''''''''''''''''
###ONLY C###
''''''''''''''''''''
''''''''''''''''''''
''''''''''''''''''''
''''''''''''''''''''

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Utility Functions
# -----------------------------
def noisy_handoff(seed_array, L):
    return seed_array  # No noise applied when L is ignored

def create_updated_array(N, K, initial_seed=None): #Rule-Based Adaptation of AI
    if K <= 0:
        return [0] * N
    if initial_seed is None or len(initial_seed) != K:
        initial_seed = [random.choice([0, 1]) for _ in range(K)]
    result = list(initial_seed)
    while len(result) < N:
        window = result[-K:]
        result.append(1 if sum(window) / K >= 0.5 else 0)
    return result

def create_updated_array_human(N, K, initial_seed=None): #Heuristic-Based Adaptation of Human
    if K <= 0:
        return [0] * N
    if initial_seed is None or len(initial_seed) != K:
        initial_seed = [random.choice([0, 1]) for _ in range(K)]
    result = list(initial_seed)
    weights = list(range(1, K + 1))
    total_weight = sum(weights)
    while len(result) < N:
        window = result[-K:]
        weighted_sum = sum(w * b for w, b in zip(weights, window))
        result.append(1 if weighted_sum >= total_weight / 2 else 0)
    return result

def compute_fitness(array, K): 
    N = len(array)
    total = 0
    for i in range(N):
        indices = [(i + j) % N for j in range(K + 1)]
        values = [array[j] for j in indices]
        total += sum(values) % 2
    return total / N

# -----------------------------
# Model Functions (L ignored)
# -----------------------------
def model_modular(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI):
    h1 = create_updated_array_human(N_H1, K_H1)
    h2 = create_updated_array_human(N_H2, K_H2)
    ai = create_updated_array(N_AI, K_AI)
    return np.mean([
        compute_fitness(h1, K_H1),
        compute_fitness(h2, K_H2),
        compute_fitness(ai, K_AI)
    ])

def model_H1_H2_AI(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, C):
    h1 = create_updated_array_human(N_H1, K_H1)
    seed_h2 = h1[:C] if C > 0 else None
    h2 = create_updated_array_human(N_H2, C if C > 0 else K_H2, seed_h2)
    seed_ai = h2[:C] if C > 0 else None
    ai = create_updated_array(N_AI, C if C > 0 else K_AI, seed_ai)
    return np.mean([
        compute_fitness(h1, K_H1),
        compute_fitness(h2, K_H2),
        compute_fitness(ai, K_AI)
    ])

def model_H1_AI_H2(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, C):
    h1 = create_updated_array_human(N_H1, K_H1)
    seed_ai = h1[:C] if C > 0 else None
    ai = create_updated_array(N_AI, C if C > 0 else K_AI, seed_ai)
    seed_h2 = ai[:C] if C > 0 else None
    h2 = create_updated_array_human(N_H2, C if C > 0 else K_H2, seed_h2)
    return np.mean([
        compute_fitness(h1, K_H1),
        compute_fitness(ai, K_AI),
        compute_fitness(h2, K_H2)
    ])

def model_AI_H1_H2(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, C):
    ai = create_updated_array(N_AI, K_AI)
    seed_h1 = ai[:C] if C > 0 else None
    h1 = create_updated_array_human(N_H1, C if C > 0 else K_H1, seed_h1)
    seed_h2 = h1[:C] if C > 0 else None
    h2 = create_updated_array_human(N_H2, C if C > 0 else K_H2, seed_h2)
    return np.mean([
        compute_fitness(ai, K_AI),
        compute_fitness(h1, K_H1),
        compute_fitness(h2, K_H2)
    ])

# -----------------------------
# Run Simulations across C (L ignored)
# -----------------------------
N_H1 = N_H2 = 10
K_H1 = K_H2 = 3
N_AI = 2 * N_H1
K_AI = 1 + K_H1
repeats = 1000
c_values = list(range(0, 11))  # C from 0 to 10
results = []

# Generate Modular average PO （fixed）
modular_po_list = [model_modular(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI) for _ in range(repeats)]
modular_mean_po = np.mean(modular_po_list)

# Add C
for c in c_values:
    for _ in range(repeats):
        # C is fixed in Modular
        results.append({'Model': 'Modular', 'C': c, 'PO': modular_mean_po})
        results.append({'Model': 'H1-H2-AI', 'C': c, 'PO': model_H1_H2_AI(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, c)})
        results.append({'Model': 'H1-AI-H2', 'C': c, 'PO': model_H1_AI_H2(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, c)})
        results.append({'Model': 'AI-H1-H2', 'C': c, 'PO': model_AI_H1_H2(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, c)})

# Plotting
df = pd.DataFrame(results)
summary_df = df.groupby(['Model', 'C'])['PO'].agg(['mean', 'std']).reset_index()

# Plotting
plt.figure(figsize=(10, 6))
for model in summary_df['Model'].unique():
    subset = summary_df[summary_df['Model'] == model]
    plt.plot(subset['C'], subset['mean'], label=model)

plt.title("PO vs C for Different Models (L Ignored)\n(N_H1 = N_H2 = 10, K_H1 = K_H2 = 3, N_AI = 2 * N_H1, K_AI = 1 + K_H1, repeats = 1000)")
plt.xlabel("Handoff Size (C)")
plt.ylabel("Average PO")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

''''''''''''''''''''
''''''''''''''''''''
''''''''''''''''''''
''''''''''''''''''''
###ONLY L, fix C###
#Having L between H and AI
''''''''''''''''''''
''''''''''''''''''''
''''''''''''''''''''
''''''''''''''''''''
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Utility Functions
# -----------------------------
def noisy_handoff(seed_array, L):
    # Each bit in the seed array has a probability L of being flipped (representing noise)
    return [1 - bit if random.random() < L else bit for bit in seed_array]

def create_updated_array(N, K, initial_seed=None):
    # AI agent's decision logic (simple majority over K-window)
    if K <= 0:
        return [0] * N
    if initial_seed is None or len(initial_seed) != K:
        initial_seed = [random.choice([0, 1]) for _ in range(K)]
    result = list(initial_seed)
    while len(result) < N:
        window = result[-K:]
        result.append(1 if sum(window) / K >= 0.5 else 0)
    return result

def create_updated_array_human(N, K, initial_seed=None):
    # Human agent's decision logic (weighted toward recent decisions)
    if K <= 0:
        return [0] * N
    if initial_seed is None or len(initial_seed) != K:
        initial_seed = [random.choice([0, 1]) for _ in range(K)]
    result = list(initial_seed)
    weights = list(range(1, K + 1))
    total_weight = sum(weights)
    while len(result) < N:
        window = result[-K:]
        weighted_sum = sum(w * b for w, b in zip(weights, window))
        result.append(1 if weighted_sum >= total_weight / 2 else 0)
    return result

def compute_fitness(array, K):
    # Fitness = proportion of parity matches over sliding windows
    N = len(array)
    total = 0
    for i in range(N):
        indices = [(i + j) % N for j in range(K + 1)]
        values = [array[j] for j in indices]
        total += sum(values) % 2
    return total / N

# -----------------------------
# Model Functions (fixed C, varying L)
# -----------------------------
C_fixed = 5  # Fixed handoff size

def model_modular(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI):
    # No coordination: agents work independently
    h1 = create_updated_array_human(N_H1, K_H1)
    h2 = create_updated_array_human(N_H2, K_H2)
    ai = create_updated_array(N_AI, K_AI)
    return np.mean([
        compute_fitness(h1, K_H1),
        compute_fitness(h2, K_H2),
        compute_fitness(ai, K_AI)
    ])

def model_H1_H2_AI(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, L):
    # Sequential: H1 → H2 → AI with noise L applied at each handoff
    h1 = create_updated_array_human(N_H1, K_H1)
    seed_h2 = noisy_handoff(h1[:C_fixed], L)
    h2 = create_updated_array_human(N_H2, C_fixed, seed_h2)
    seed_ai = h2[:C_fixed] #No L between H and AI
    ai = create_updated_array(N_AI, C_fixed, seed_ai)
    return np.mean([
        compute_fitness(h1, K_H1),
        compute_fitness(h2, K_H2),
        compute_fitness(ai, K_AI)
    ])

def model_H1_AI_H2(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, L):
    # Sequential: H1 → AI → H2 with noisy handoffs
    h1 = create_updated_array_human(N_H1, K_H1)
    seed_ai = noisy_handoff(h1[:C_fixed], L)
    ai = create_updated_array(N_AI, C_fixed, seed_ai)
    seed_h2 = noisy_handoff(ai[:C_fixed], L)
    h2 = create_updated_array_human(N_H2, C_fixed, seed_h2)
    return np.mean([
        compute_fitness(h1, K_H1),
        compute_fitness(ai, K_AI),
        compute_fitness(h2, K_H2)
    ])

def model_AI_H1_H2(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, L):
    # Sequential: AI → H1 → H2 with noisy handoffs
    ai = create_updated_array(N_AI, K_AI)
    seed_h1 = noisy_handoff(ai[:C_fixed], L)
    h1 = create_updated_array_human(N_H1, C_fixed, seed_h1)
    seed_h2 = noisy_handoff(h1[:C_fixed], L)
    h2 = create_updated_array_human(N_H2, C_fixed, seed_h2)
    return np.mean([
        compute_fitness(ai, K_AI),
        compute_fitness(h1, K_H1),
        compute_fitness(h2, K_H2)
    ])

# -----------------------------
# Run Simulations (L from 0 to 1)
# -----------------------------
N_H1 = N_H2 = 10
K_H1 = K_H2 = 3
N_AI = 2 * N_H1
K_AI = K_H1
repeats = 1000
l_values = np.linspace(0, 1, 11)  # L = 0.0 to 1.0

results = []

# Precompute modular model average PO once (fixed value across all L)
modular_po_list = [model_modular(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI) for _ in range(repeats)]
modular_mean_po = np.mean(modular_po_list)

for l in l_values:
    for _ in range(repeats):
        # Use constant modular mean PO for every L value
        results.append({'Model': 'Modular', 'L': l, 'PO': modular_mean_po})
        results.append({'Model': 'H1-H2-AI', 'L': l, 'PO': model_H1_H2_AI(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, l)})
        results.append({'Model': 'H1-AI-H2', 'L': l, 'PO': model_H1_AI_H2(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, l)})
        results.append({'Model': 'AI-H1-H2', 'L': l, 'PO': model_AI_H1_H2(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, l)})

# -----------------------------
# Aggregate Results and Plot
# -----------------------------
df = pd.DataFrame(results)
summary_df = df.groupby(['Model', 'L'])['PO'].agg(['mean', 'std']).reset_index()

plt.figure(figsize=(10, 6))
for model in summary_df['Model'].unique():
    subset = summary_df[summary_df['Model'] == model]
    plt.plot(subset['L'], subset['mean'], label=model)

plt.title("PO vs L for Different Models \n(Noise among H1, H2, and AI, C = 5)\n(N_H1 = N_H2 = 10, K_H1 = K_H2 = 3, N_AI = 2 * N_H1, K_AI = K_H1, repeats = 1000)")
plt.xlabel("Coordination Noise (L)")
plt.ylabel("Average PO")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


''''''''''''''''''''
''''''''''''''''''''
''''''''''''''''''''
''''''''''''''''''''
###ONLY L, fix C###
#No L between H and AI
''''''''''''''''''''
''''''''''''''''''''
''''''''''''''''''''
''''''''''''''''''''

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Utility Functions
# -----------------------------
def noisy_handoff(seed_array, L):
    return [1 - bit if random.random() < L else bit for bit in seed_array]

def create_updated_array(N, K, initial_seed=None):
    if K <= 0:
        return [0] * N
    if initial_seed is None or len(initial_seed) != K:
        initial_seed = [random.choice([0, 1]) for _ in range(K)]
    result = list(initial_seed)
    while len(result) < N:
        window = result[-K:]
        result.append(1 if sum(window) / K >= 0.5 else 0)
    return result

def create_updated_array_human(N, K, initial_seed=None):
    if K <= 0:
        return [0] * N
    if initial_seed is None or len(initial_seed) != K:
        initial_seed = [random.choice([0, 1]) for _ in range(K)]
    result = list(initial_seed)
    weights = list(range(1, K + 1))
    total_weight = sum(weights)
    while len(result) < N:
        window = result[-K:]
        weighted_sum = sum(w * b for w, b in zip(weights, window))
        result.append(1 if weighted_sum >= total_weight / 2 else 0)
    return result

def compute_fitness(array, K):
    N = len(array)
    total = 0
    for i in range(N):
        indices = [(i + j) % N for j in range(K + 1)]
        values = [array[j] for j in indices]
        total += sum(values) % 2
    return total / N

# -----------------------------
# Model Functions
# -----------------------------
C_fixed = 5

def model_modular(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI):
    h1 = create_updated_array_human(N_H1, K_H1)
    h2 = create_updated_array_human(N_H2, K_H2)
    ai = create_updated_array(N_AI, K_AI)
    return np.mean([
        compute_fitness(h1, K_H1),
        compute_fitness(h2, K_H2),
        compute_fitness(ai, K_AI)
    ])

def model_H1_H2_AI(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, L):
    h1 = create_updated_array_human(N_H1, K_H1)
    seed_h2 = noisy_handoff(h1[:C_fixed], L)
    h2 = create_updated_array_human(N_H2, C_fixed, seed_h2)
    seed_ai = h2[:C_fixed]  # No noise
    ai = create_updated_array(N_AI, C_fixed, seed_ai)
    return np.mean([
        compute_fitness(h1, K_H1),
        compute_fitness(h2, K_H2),
        compute_fitness(ai, K_AI)
    ])

def model_H1_AI_H2(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI):
    h1 = create_updated_array_human(N_H1, K_H1)
    seed_ai = h1[:C_fixed]  # No noise
    ai = create_updated_array(N_AI, C_fixed, seed_ai)
    seed_h2 = ai[:C_fixed]  # No noise
    h2 = create_updated_array_human(N_H2, C_fixed, seed_h2)
    return np.mean([
        compute_fitness(h1, K_H1),
        compute_fitness(ai, K_AI),
        compute_fitness(h2, K_H2)
    ])

def model_AI_H1_H2(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, L):
    ai = create_updated_array(N_AI, K_AI)
    seed_h1 = ai[:C_fixed]  # No noise
    h1 = create_updated_array_human(N_H1, C_fixed, seed_h1)
    seed_h2 = noisy_handoff(h1[:C_fixed], L)
    h2 = create_updated_array_human(N_H2, C_fixed, seed_h2)
    return np.mean([
        compute_fitness(ai, K_AI),
        compute_fitness(h1, K_H1),
        compute_fitness(h2, K_H2)
    ])

# -----------------------------
# Run Simulations
# -----------------------------
N_H1 = N_H2 = 10
K_H1 = K_H2 = 3
N_AI = 2 * N_H1
K_AI = K_H1
repeats = 1000
l_values = np.linspace(0, 1, 11)

results = []

modular_po_list = [model_modular(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI) for _ in range(repeats)]
modular_mean_po = np.mean(modular_po_list)

for l in l_values:
    for _ in range(repeats):
        results.append({'Model': 'Modular', 'L': l, 'PO': modular_mean_po})
        results.append({'Model': 'H1-H2-AI', 'L': l, 'PO': model_H1_H2_AI(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, l)})
        results.append({'Model': 'H1-AI-H2', 'L': l, 'PO': model_H1_AI_H2(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI)})
        results.append({'Model': 'AI-H1-H2', 'L': l, 'PO': model_AI_H1_H2(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, l)})

# -----------------------------
# Plot Results
# -----------------------------
df = pd.DataFrame(results)
summary_df = df.groupby(['Model', 'L'])['PO'].agg(['mean', 'std']).reset_index()

plt.figure(figsize=(10, 6))
for model in summary_df['Model'].unique():
    subset = summary_df[summary_df['Model'] == model]
    plt.plot(subset['L'], subset['mean'], label=model)

plt.title("PO vs Coordination Noise (L) for Different Models\n(Noise Only Between H1 and H2, C = 5)\n(N_H1 = N_H2 = 10, K_H1 = K_H2 = 3, N_AI = 2 * N_H1, K_AI = K_H1, repeats = 1000)")
plt.xlabel("Coordination Noise (L)")
plt.ylabel("Average Performance Outcome (PO)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


''''''''''''''''''''
''''''''''''''''''''
''''''''''''''''''''
''''''''''''''''''''
###C and L###
###L among all agents###
''''''''''''''''''''
''''''''''''''''''''
''''''''''''''''''''
''''''''''''''''''''
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Utility Functions
def noisy_handoff(seed_array, L):
    return [1 - bit if random.random() < L else bit for bit in seed_array]

def create_updated_array(N, K, initial_seed=None):
    if K <= 0:
        return [0] * N
    if initial_seed is None or len(initial_seed) != K:
        initial_seed = [random.choice([0, 1]) for _ in range(K)]
    result = list(initial_seed)
    while len(result) < N:
        window = result[-K:]
        result.append(1 if sum(window) / K >= 0.5 else 0)
    return result

def create_updated_array_human(N, K, initial_seed=None):
    if K <= 0:
        return [0] * N
    if initial_seed is None or len(initial_seed) != K:
        initial_seed = [random.choice([0, 1]) for _ in range(K)]
    result = list(initial_seed)
    weights = list(range(1, K + 1))
    total_weight = sum(weights)
    while len(result) < N:
        window = result[-K:]
        weighted_sum = sum(w * b for w, b in zip(weights, window))
        result.append(1 if weighted_sum >= total_weight / 2 else 0)
    return result

def compute_fitness(array, K):
    N = len(array)
    total = 0
    for i in range(N):
        indices = [(i + j) % N for j in range(K + 1)]
        values = [array[j] for j in indices]
        total += sum(values) % 2
    return total / N

# Model Functions
def model_modular(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI):
    h1 = create_updated_array_human(N_H1, K_H1)
    h2 = create_updated_array_human(N_H2, K_H2)
    ai = create_updated_array(N_AI, K_AI)
    return np.mean([
        compute_fitness(h1, K_H1),
        compute_fitness(h2, K_H2),
        compute_fitness(ai, K_AI)
    ])

def model_H1_H2_AI(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, L, C):
    h1 = create_updated_array_human(N_H1, K_H1)
    seed_h2 = noisy_handoff(h1[:C], L) if C > 0 else None
    h2 = create_updated_array_human(N_H2, C if C > 0 else K_H2, seed_h2)
    seed_ai = noisy_handoff(h2[:C], L) if C > 0 else None  # <--- Noise
    ai = create_updated_array(N_AI, C if C > 0 else K_AI, seed_ai)
    return np.mean([
        compute_fitness(h1, K_H1),
        compute_fitness(h2, K_H2),
        compute_fitness(ai, K_AI)
    ])

def model_H1_AI_H2(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, L, C):
    h1 = create_updated_array_human(N_H1, K_H1)
    seed_ai = noisy_handoff(h1[:C], L) if C > 0 else None
    ai = create_updated_array(N_AI, C if C > 0 else K_AI, seed_ai)
    seed_h2 = noisy_handoff(ai[:C], L) if C > 0 else None
    h2 = create_updated_array_human(N_H2, C if C > 0 else K_H2, seed_h2)
    return np.mean([
        compute_fitness(h1, K_H1),
        compute_fitness(ai, K_AI),
        compute_fitness(h2, K_H2)
    ])

def model_AI_H1_H2(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, L, C):
    ai = create_updated_array(N_AI, K_AI)
    seed_h1 = noisy_handoff(ai[:C], L) if C > 0 else None  # <--- Noise
    h1 = create_updated_array_human(N_H1, C if C > 0 else K_H1, seed_h1)
    seed_h2 = noisy_handoff(h1[:C], L) if C > 0 else None  
    h2 = create_updated_array_human(N_H2, C if C > 0 else K_H2, seed_h2)
    return np.mean([
        compute_fitness(ai, K_AI),
        compute_fitness(h1, K_H1),
        compute_fitness(h2, K_H2)
    ])

# Parameters
N_H1 = N_H2 = 10
K_H1 = K_H2 = 3
N_AI = 2 * N_H1
K_AI = K_H1
C_values = list(range(0, 11))
L_values = np.linspace(0, 1, 11) #np.linspace(start, stop, num)
repeats = 1000

# Simulation
results = []
modular_results = [model_modular(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI) for _ in range(repeats)]
modular_mean_po = np.mean(modular_results)

for C in C_values:
    for L in L_values:
        for _ in range(repeats):
            results.append({'Model': 'H1-H2-AI', 'C': C, 'L': L,
                            'PO': model_H1_H2_AI(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, L, C)})
            results.append({'Model': 'H1-AI-H2', 'C': C, 'L': L,
                            'PO': model_H1_AI_H2(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, L, C)})
            results.append({'Model': 'AI-H1-H2', 'C': C, 'L': L,
                            'PO': model_AI_H1_H2(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, L, C)})

df = pd.DataFrame(results)
df_avg = df.groupby(['Model', 'C', 'L'])['PO'].mean().reset_index()

# Plotting
po_min = df_avg['PO'].min()
po_max = df_avg['PO'].max()

models = df_avg['Model'].unique()
fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

for idx, model in enumerate(models):
    ax = axes[idx]
    subset = df_avg[df_avg['Model'] == model]
    contour_data = subset.pivot(index='L', columns='C', values='PO')
    C_vals = contour_data.columns.values
    L_vals = contour_data.index.values
    C_mesh, L_mesh = np.meshgrid(C_vals, L_vals)
    Z = contour_data.values

    contour = ax.contourf(C_mesh, L_mesh, Z, levels=20, cmap="viridis", vmin=po_min, vmax=po_max)
    lines = ax.contour(C_mesh, L_mesh, Z, levels=10, colors='black', linewidths=0.5)
    ax.clabel(lines, inline=True, fontsize=8, fmt="%.2f")

    ax.contour(C_mesh, L_mesh, Z, levels=[modular_mean_po], colors='red', linestyles='dashed', linewidths=1.5)
    ax.text(C_vals[-1], L_vals[-1], f'Modular PO ≈ {modular_mean_po:.2f}', color='red', fontsize=9,
            verticalalignment='top', horizontalalignment='right')

    ax.set_title(f"PO Contour: {model}")
    ax.set_xlabel("C (Handoff Size)")
    if idx == 0:
        ax.set_ylabel("L (Coordination Noise)")

# Adjust layout to make room for colorbar
fig.subplots_adjust(right=0.88)  # Shift content left to make room for colorbar
cbar_ax = fig.add_axes([1.0, 0.09, 0.015, 0.7])  # Position: [left, bottom, width, height]
fig.colorbar(contour, cax=cbar_ax, label="Average PO")
plt.suptitle("PO Contour with L among all Agents \n(N_H1 = N_H2 = 10, K_H1 = K_H2 = 3, N_AI = 2 * N_H1, K_AI = K_H1, repeats = 1000)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()




''''''''''''''''''''
''''''''''''''''''''
''''''''''''''''''''
''''''''''''''''''''
###C and L###
###L only between H1 and H2###
''''''''''''''''''''
''''''''''''''''''''
''''''''''''''''''''
# Re-import required packages after reset
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Utility Functions
def noisy_handoff(seed_array, L):
    return [1 - bit if random.random() < L else bit for bit in seed_array]

def create_updated_array(N, K, initial_seed=None):
    if K <= 0:
        return [0] * N
    if initial_seed is None or len(initial_seed) != K:
        initial_seed = [random.choice([0, 1]) for _ in range(K)]
    result = list(initial_seed)
    while len(result) < N:
        window = result[-K:]
        result.append(1 if sum(window) / K >= 0.5 else 0)
    return result

def create_updated_array_human(N, K, initial_seed=None):
    if K <= 0:
        return [0] * N
    if initial_seed is None or len(initial_seed) != K:
        initial_seed = [random.choice([0, 1]) for _ in range(K)]
    result = list(initial_seed)
    weights = list(range(1, K + 1))
    total_weight = sum(weights)
    while len(result) < N:
        window = result[-K:]
        weighted_sum = sum(w * b for w, b in zip(weights, window))
        result.append(1 if weighted_sum >= total_weight / 2 else 0)
    return result

def compute_fitness(array, K):
    N = len(array)
    total = 0
    for i in range(N):
        indices = [(i + j) % N for j in range(K + 1)]
        values = [array[j] for j in indices]
        total += sum(values) % 2
    return total / N

# Model Functions (L only between H1 and H2)
def model_modular(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI):
    h1 = create_updated_array_human(N_H1, K_H1)
    h2 = create_updated_array_human(N_H2, K_H2)
    ai = create_updated_array(N_AI, K_AI)
    return np.mean([
        compute_fitness(h1, K_H1),
        compute_fitness(h2, K_H2),
        compute_fitness(ai, K_AI)
    ])

def model_H1_H2_AI(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, L, C):
    h1 = create_updated_array_human(N_H1, K_H1)
    seed_h2 = noisy_handoff(h1[:C], L) if C > 0 else None
    h2 = create_updated_array_human(N_H2, C if C > 0 else K_H2, seed_h2)
    seed_ai = h2[:C] if C > 0 else None  # No noise here
    ai = create_updated_array(N_AI, C if C > 0 else K_AI, seed_ai)
    return np.mean([
        compute_fitness(h1, K_H1),
        compute_fitness(h2, K_H2),
        compute_fitness(ai, K_AI)
    ])

def model_H1_AI_H2(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, L, C):
    h1 = create_updated_array_human(N_H1, K_H1)
    seed_ai = h1[:C] if C > 0 else None  # No noise here
    ai = create_updated_array(N_AI, C if C > 0 else K_AI, seed_ai)
    seed_h2 = noisy_handoff(ai[:C], L) if C > 0 else None
    h2 = create_updated_array_human(N_H2, C if C > 0 else K_H2, seed_h2)
    return np.mean([
        compute_fitness(h1, K_H1),
        compute_fitness(ai, K_AI),
        compute_fitness(h2, K_H2)
    ])

def model_AI_H1_H2(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, L, C):
    ai = create_updated_array(N_AI, K_AI)
    seed_h1 = ai[:C] if C > 0 else None  # No noise here
    h1 = create_updated_array_human(N_H1, C if C > 0 else K_H1, seed_h1)
    seed_h2 = noisy_handoff(h1[:C], L) if C > 0 else None
    h2 = create_updated_array_human(N_H2, C if C > 0 else K_H2, seed_h2)
    return np.mean([
        compute_fitness(ai, K_AI),
        compute_fitness(h1, K_H1),
        compute_fitness(h2, K_H2)
    ])

# Parameters
N_H1 = N_H2 = 10
K_H1 = K_H2 = 3
N_AI = 2 * N_H1
K_AI = K_H1
C_values = list(range(0, 11))
L_values = np.linspace(0, 1, 11) #np.linspace(start, stop, num)
repeats = 1000

# Simulation
results = []
modular_results = [model_modular(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI) for _ in range(repeats)]
modular_mean_po = np.mean(modular_results)

for C in C_values:
    for L in L_values:
        for _ in range(repeats):
            results.append({'Model': 'H1-H2-AI', 'C': C, 'L': L,
                            'PO': model_H1_H2_AI(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, L, C)})
            results.append({'Model': 'H1-AI-H2', 'C': C, 'L': L,
                            'PO': model_H1_AI_H2(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, L, C)})
            results.append({'Model': 'AI-H1-H2', 'C': C, 'L': L,
                            'PO': model_AI_H1_H2(N_H1, K_H1, N_H2, K_H2, N_AI, K_AI, L, C)})

df = pd.DataFrame(results)
df_avg = df.groupby(['Model', 'C', 'L'])['PO'].mean().reset_index()

# Plotting
po_min = df_avg['PO'].min()
po_max = df_avg['PO'].max()

models = df_avg['Model'].unique()
fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

for idx, model in enumerate(models):
    ax = axes[idx]
    subset = df_avg[df_avg['Model'] == model]
    contour_data = subset.pivot(index='L', columns='C', values='PO')
    C_vals = contour_data.columns.values
    L_vals = contour_data.index.values
    C_mesh, L_mesh = np.meshgrid(C_vals, L_vals)
    Z = contour_data.values

    contour = ax.contourf(C_mesh, L_mesh, Z, levels=20, cmap="viridis", vmin=po_min, vmax=po_max)
    lines = ax.contour(C_mesh, L_mesh, Z, levels=10, colors='black', linewidths=0.5)
    ax.clabel(lines, inline=True, fontsize=8, fmt="%.2f")

    ax.contour(C_mesh, L_mesh, Z, levels=[modular_mean_po], colors='red', linestyles='dashed', linewidths=1.5)
    ax.text(C_vals[-1], L_vals[-1], f'Modular PO ≈ {modular_mean_po:.2f}', color='red', fontsize=9,
            verticalalignment='top', horizontalalignment='right')

    ax.set_title(f"PO Contour: {model}")
    ax.set_xlabel("C (Handoff Size)")
    if idx == 0:
        ax.set_ylabel("L (Coordination Noise)")

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([1.0, 0.09, 0.015, 0.7])
fig.colorbar(contour, cax=cbar_ax, label="Average PO")
plt.suptitle("PO Contour with L Only Between H1 and H2 \n(N_H1 = N_H2 = 10, K_H1 = K_H2 = 3, N_AI = 2 * N_H1, K_AI = K_H1, repeats = 1000)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

