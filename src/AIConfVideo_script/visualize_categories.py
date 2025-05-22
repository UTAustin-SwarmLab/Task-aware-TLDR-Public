import matplotlib.pyplot as plt
import squarify

# Data
categories = {
    "Graph-Based & Neurosymbolic Learning": 95,
    "Reinforcement Learning & Robotics": 260,
    "Representation Learning": 89,
    "Infrastructure, Benchmarks & Evaluation": 77,
    "Optimization": 200,
    "Generative Models": 265,
    "Learning Theory": 334,
    "Probabilistic & Causal Methods": 149,
    "Human-AI Interaction and Ethics": 241,
    "Applications to Sciences & Engineering": 254,
    "Natural Language, Vision & Multimodal Learning": 463,
    "Others": 129,
}

# Sort categories by count
sorted_categories = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))

# Prepare data for squarify
sizes = list(sorted_categories.values())


# Wrap labels to fit in boxes
def wrap_label(text, count, max_chars=20):
    # Split the text into words
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_chars:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(" ".join(current_line))

    # Add count to the last line
    lines[-1] = f"{lines[-1]}\n({count})"
    return "\n".join(lines)


labels = [wrap_label(k, v) for k, v in sorted_categories.items()]

# Colorblind-friendly color scheme (ColorBrewer)
colors = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928",
]

# Create figure
plt.figure(figsize=(15, 10))

# Create treemap without padding
squarify.plot(
    sizes=sizes,
    label=labels,
    color=colors,
    alpha=0.8,
    text_kwargs={"fontsize": 16, "fontweight": "bold", "color": "black", "wrap": True},
    pad=False,  # Remove padding
    #   bar_kwargs={'linewidth': 1, 'edgecolor': 'black'}# Black borders
)

# Remove axes
plt.axis("off")

# Add title
plt.title("Paper Categories Distribution", pad=20, fontsize=16, fontweight="bold")

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig("paper_categories_treemap.png", dpi=300, bbox_inches="tight")
plt.close()
