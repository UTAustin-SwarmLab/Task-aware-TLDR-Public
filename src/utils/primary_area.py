import json

subarea_map = {
    "unsupervised, self-supervised, semi-supervised, and supervised representation learning": "Representation Learning",
    "visualization or interpretation of learned representations": "Representation Learning",
    "reinforcement learning": "Reinforcement Learning & Robotics",
    "robotics": "Reinforcement Learning & Robotics",
    "applications to robotics, autonomy, planning": "Reinforcement Learning & Robotics",
    "learning theory": "Learning Theory",
    "metric learning, kernel learning, and sparse coding": "Learning Theory",
    "deep learning architectures": "Learning Theory",
    "online learning": "Learning Theory",
    "active learning": "Learning Theory",
    "bandits": "Learning Theory",
    "transfer learning, meta learning, and lifelong learning": "Learning Theory",
    "probabilistic methods": "Probabilistic & Causal Methods",
    "probabilistic methods (bayesian methods, variational inference, sampling, uq, etc.)": "Probabilistic & Causal Methods",
    "causal reasoning": "Probabilistic & Causal Methods",
    "causal inference": "Probabilistic & Causal Methods",
    "learning on graphs and other geometries & topologies": "Graph-Based & Neurosymbolic Learning",
    "graph neural networks": "Graph-Based & Neurosymbolic Learning",
    "neurosymbolic & hybrid ai systems (physics-informed, logic & formal reasoning, etc.)": "Graph-Based & Neurosymbolic Learning",
    "applications to physical sciences (physics, chemistry, biology, etc.)": "Applications to Sciences & Engineering",
    "applications to neuroscience & cognitive science": "Applications to Sciences & Engineering",
    "machine learning for physical sciences": "Applications to Sciences & Engineering",
    "machine learning for other sciences and fields": "Applications to Sciences & Engineering",
    "neuroscience and cognitive science": "Applications to Sciences & Engineering",
    "machine learning for social sciences": "Applications to Sciences & Engineering",
    "machine learning for healthcare": "Applications to Sciences & Engineering",
    "fairness": "Human-AI Interaction and Ethics (Privacy, Fairness & Safety)",
    "safety in machine learning": "Human-AI Interaction and Ethics (Privacy, Fairness & Safety)",
    "privacy": "Human-AI Interaction and Ethics (Privacy, Fairness & Safety)",
    "societal considerations including fairness, safety, privacy": "Human-AI Interaction and Ethics (Privacy, Fairness & Safety)",
    "human-ai interaction": "Human-AI Interaction and Ethics (Privacy, Fairness & Safety)",
    "interpretability and explainability": "Human-AI Interaction and Ethics (Privacy, Fairness & Safety)",
    "representation learning for computer vision": "Natural Language, Vision & Multimodal Learning",
    "audio": "Natural Language, Vision & Multimodal Learning",
    "language": "Natural Language, Vision & Multimodal Learning",
    "natural language processing": "Natural Language, Vision & Multimodal Learning",
    "speech and audio": "Natural Language, Vision & Multimodal Learning",
    "machine vision": "Natural Language, Vision & Multimodal Learning",
    "representation learning for computer vision, audio, language, and other modalities": "Natural Language, Vision & Multimodal Learning",
    "infrastructure": "Infrastructure, Benchmarks & Evaluation",
    "infrastructure, software libraries, hardware, etc.": "Infrastructure, Benchmarks & Evaluation",
    "datasets and benchmarks": "Infrastructure, Benchmarks & Evaluation",
    "evaluation": "Infrastructure, Benchmarks & Evaluation",
    "optimization": "Optimization",
    "algorithmic game theory": "Optimization",
    "diffusion based models": "Generative Models",
    "generative models": "Generative Models",
    "general machine learning (i.e., none of the above)": "Others",
    "other": "Others",
    "optimization for deep networks": "Others",
}


def MapArea2Category(subarea):
    # Look up the subarea
    category = subarea_map.get(subarea.lower(), "Unknown Category")
    if category == "Unknown Category":
        print("=========================================================")
        print(f"Unknown subarea: {subarea}")
        print("=========================================================")
    return category


def CntPrimaryArea(
    json_paths: list[str] = [
        "/nas/pohan/datasets/AIConfVideo/dataset/paperlist_iclr2024.json",
        "/nas/pohan/datasets/AIConfVideo/dataset/paperlist_nips2024.json",
    ],
) -> dict:
    area2cnt = {}
    # Load JSON files
    for json_path in json_paths:
        conf_area2cnt = {}
        with open(json_path, "r") as file:
            data = json.load(file)
        for paper in data:
            area = paper["primary_area"].replace("_", " ").lower()
            if area not in conf_area2cnt:
                conf_area2cnt[area] = 0
            conf_area2cnt[area] += 1

        print(conf_area2cnt, "\n", len(conf_area2cnt))
        for area, cnt in conf_area2cnt.items():
            if area not in area2cnt:
                area2cnt[area] = 0
            area2cnt[area] += cnt

    print(area2cnt, "\n", len(area2cnt))
    return area2cnt


if __name__ == "__main__":
    area2cnt = CntPrimaryArea()
    cate2cnt = {}
    for area in area2cnt.keys():
        cate = MapArea2Category(area)
        if cate not in cate2cnt:
            cate2cnt[cate] = 0
        cate2cnt[cate] += area2cnt[area]
    print(cate2cnt)


"""
1. **Representation Learning** 
   - ICLR: 'unsupervised, self-supervised, semi-supervised, and supervised representation learning' (73), 'visualization or interpretation of learned representations' (16)  

2. **Reinforcement Learning & Robotics**  
   - ICLR: 'reinforcement learning' (87), 'applications to robotics, autonomy, planning' (16)  
   - NeurIPS: 'reinforcement learning' (136), 'robotics' (21) 

3. **Learning Theory**  
   - ICLR: 'learning theory' (29), 'metric learning, kernel learning, and sparse coding' (4)  , 'transfer learning, meta learning, and lifelong learning'
   - NeurIPS: 'deep learning architectures' (86), 'learning theory' (124), 'online learning' (16), 'active learning' (13), 'bandits' (32)

4. **Probabilistic & Causal Methods**  
   - ICLR: 'probabilistic methods (Bayesian methods, variational inference, sampling, UQ, etc.)' (28), 'causal reasoning' (7)  
   - NeurIPS: 'probabilistic methods' (78), 'causal inference' (36)  

5. **Graph-Based & Neurosymbolic Learning**  
   - ICLR: 'learning on graphs and other geometries & topologies' (24), 'neurosymbolic & hybrid AI systems (physics-informed, logic & formal reasoning, etc.)' (14)  
   - NeurIPS: 'graph neural networks' (57)  

6. **Applications to Sciences & Engineering**  
   - ICLR: 'applications to physical sciences (physics, chemistry, biology, etc.)' (35), 'applications to neuroscience & cognitive science' (16)  
   - NeurIPS: 'machine learning for physical sciences' (39), 'machine learning for other sciences and fields' (56), 'neuroscience and cognitive science' (59)  

7. **Human-AI Interaction and Ethics (Privacy, Fairness & Safety)**
   - ICLR: 'societal considerations including fairness, safety, privacy' (49)  
   - NeurIPS: 'fairness' (20), 'safety in machine learning' (63), 'privacy' (43), 'human-AI interaction' (11)  

8. **Natural Language, Vision & Multimodal Learning**  
   - ICLR:  'representation learning for computer vision, audio, language, and other modalities' (79)
   - NeurIPS: 'natural language processing' (131), 'speech and audio' (20), 'machine vision' (233)  

9. **Infrastructure, Benchmarks & Evaluation**  
   - ICLR: 'infrastructure, software libraries, hardware, etc.' (5), 'datasets and benchmarks' (29)  
   - NeurIPS: 'infrastructure' (18), 'evaluation' (25)  

10. **Optimization**
    - ICLR: 'optimization' (34)
    - NeurIPS: 'optimization for deep networks' (50), 'optimization' (89), 'algorithmic game theory' (27)

11. **Generative Models** 
   - NeurIPS: 'diffusion based models' (91), 'generative models' (83)

12. **Others**
"""
