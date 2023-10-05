
# on swarm intelligence algorithms
## reflection
it looks like the algorithms are more about finding a solution, f.e. genetic algorithms, instead of learning a strategy of an agent.

## ant colony optimization
ACO algorithms can handle complex situations efficiently, such as optimizing routes for transporting goods or finding the shortest path between points.
In ACO, ants communicate indirectly through a chemical substance called pheromone, which they deposit during their movement. 
Other ants follow these pheromone trails to reinforce them. 
Artificial ACO algorithms similarly involve agents communicating indirectly and following paths based on pheromone trails. 
The algorithm iteratively refines solutions by choosing paths with reinforced pheromone and discarding those with fading pheromone.

## artificial bee colony
This algorithm is derived from the behavior of bee swarms and is applied to optimization problems. 
It consists of employed, onlooker, and scout bees, each playing a role in solution search. 
Employed bees update their knowledge, which is shared with onlooker bees to find the best solution. 
Scout bees perform random searches for new solutions.

### algorithm
In ABC, a population based algorithm, the position of a food source represents a possible solution to the optimization problem and the nectar amount of a food source corresponds to the quality (fitness) of the associated solution. The number of the employed bees is equal to the number of solutions in the population. At the first step, a randomly distributed initial population (food source positions) is generated. After initialization, the population is subjected to repeat the cycles of the search processes of the employed, onlooker, and scout bees, respectively. An employed bee produces a modification on the source position in her memory and discovers a new food source position. Provided that the nectar amount of the new one is higher than that of the previous source, the bee memorizes the new source position and forgets the old one. Otherwise she keeps the position of the one in her memory. After all employed bees complete the search process, they share the position information of the sources with the onlookers on the dance area. Each onlooker evaluates the nectar information taken from all employed bees and then chooses a food source depending on the nectar amounts of sources. As in the case of the employed bee, she produces a modification on the source position in her memory and checks its nectar amount. Providing that its nectar is higher than that of the previous one, the bee memorizes the new position and forgets the old one. The sources abandoned are determined and new sources are randomly produced to be replaced with the abandoned ones by artificial scouts. 

## Comparison: Swarm Intelligence vs. Multi-Agent Reinforcement Learning

### Swarm Intelligence:

- **Decentralized:** Agents make decisions based on local information without a central controller.
- **Emergent Behavior:** Complex and adaptive global behavior arises from interactions between simple agents.
- **Often Inspired by Nature:** SI algorithms are inspired by observations of collective behaviors in nature.
- **No Explicit Communication:** Agents may not communicate directly; they influence each other by modifying the environment (e.g., pheromone trails).
- **Applications:** Optimization problems, routing and scheduling problems, clustering and pattern recognition, dynamic system control, image and data processing.

### Multi-Agent Reinforcement Learning:

- **Agents Learn:** Each agent learns its own policy by interacting with the environment and observing rewards.
- **Interaction and Adaptation:** Agents' actions impact each other's rewards and learning process, leading to adaptation.
- **Communication Possibilities:** Agents may communicate directly or indirectly to improve cooperation or coordination.
- **Trade-off between Cooperation and Competition:** Agents balance cooperation for collective success and competition for individual rewards.
- **Applications:** Autonomous vehicles and robotics, network routing and management, economic and market modeling, multi-player games.

### Comparison:

- **Communication:** In MARL, agents can often communicate explicitly, while SI relies more on indirect interactions.
- **Learning Paradigm:** MARL involves learning policies through interactions and feedback, whereas SI is often rule-based or heuristics-driven.
- **Decentralization:** Both approaches can involve decentralized decision-making, but SI is more explicitly nature-inspired.
- **Scope:** SI is more widely used in optimization and algorithmic problem-solving, while MARL is well-suited for dynamic environments and complex interactions.
