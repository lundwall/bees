
## general
swarm algorithms optimize a problem  
marl trains agents to learn a strategy.
centralised training, decentralised execution
teaching? what if one knows a good strategy?
what is the motivation of the agent? (confilcting, mixed, mutual)


## MARL
https://github.com/LantaoYu/MARL-Papers

## path towards general swarm intelligence
### reflexion
there is narrow and general swarm intelligence.
in narrow swarm intelligence, the problem is posed and the hyperparameters etc. are tuned with respect to a hand-crafted rewared function.
narrow swarm intelligence is pre-optimised, and often fails to adapt to changes in environment (it is necessary to still carry out the same task).
general swarm intelligence needs to learn how to interpret environmental cues to know when to switch tasks, with the pitfal of false alarms.  
  
common task in swarm intelligence in robotics are search and rescue, surveillance, and exploration.

### flexibility vs. adaptivity 
"flexibility" and "adaptivity" are often used interchangeably, but according to Dorigo et al. (2021), flexibility refers to a system's ability to perform tasks beyond its initial design, while adaptivity pertains to a system's capacity to learn and change its behavior in response to new operating conditions.

### definition 
definition of the concept of general swarm intelligence and address what we believe is required to achieve true adaptivity, i.e., a system’s ability to learn or change its behavior in response to new operating conditions.

Swarm Intelligence is the emergent ability of a decentralized system of agents to make the appropriate adjustments to its collective behavior, thereby allowing the system to achieve changing goals in dynamic environments.

### human vs. swarm
AI is typically defined and constructed based on human intelligence characteristics.
Natural swarm intelligence (SI) in animal groups represents a distinct, collective, and decentralized form of intelligence, separate from individual human intelligence.
To grasp SI, one can liken each bird in a flock to a neuron in a brain, with intelligence emerging from their interactions.
The current AI framework doesn't always align with SI's decentralized information gathering, social information transfer, and distributed processing.
However, there's broad scientific consensus that SI is a subset of AI, even if its precise place within AI remains undefined (Bonabeau et al., 1999; Sadiku et al., 2021).

### non-adaptive environments
Take, for example, a multirobot system tasked with retrieving and delivering packages to and from designated locations within a warehouse.
When operating within a predictable, organized, and mostly static environment, a centralized pre-planned strategy that has been optimized for a specific warehouse is often the ideal solution (Ma et al., 2017; Bredeche and Fontbonne, 2022).


## Hüttenrauch: deep rl for swarm systems

### graph abstraction
A common method to obtain control strategies for swarm systems is to apply optimization-based approaches using a model of the agents or a graph abstraction of the swarm (Lin et al., 2004; Jadbabaie et al., 2003).
Optimization-based approaches allow us to compute optimal control policies for tasks that can be well modeled, such as rendezvous or consensus problems (Lin et al., 2007) and formation control (Ranjbar-Sahraei et al., 2012), or to learn pursuit strategies to capture an evader (Zhou et al., 2016).
Yet, these approaches typically use simplified models of the agents or the task and often rely on unrealistic assumptions, such as operating in a connected graph (Dimarogonas and Kyriakopoulos, 2007) or having full observability of the system state (Zhou et al., 2016).
Rule-based approaches use heuristics inspired by natural swarm systems, such as ants or bees (Handl and Meyer, 2007).
Yet, while the resulting heuristics are often simple and can lead to complex swarm behavior, the obtained rules are difficult to adapt, even if the underlying task changes only slightly.

### swarm setting challenges
1. High state and observation dimensionality, caused by large system sizes.
2. Changing size of the available information set, either due to addition or removal of agents, or because the number of observed neighbors changes over time.


## Faust: Evolving Rewards to Automate Reinforcement Learning
### reflection
rl is heavily guided by the reward function, which is difficult and cumbersome.
autorl can be used to find an optimal reward function or to better tune hyperparameters.
it becomes increasingly interessting with complex problems. 

### defniition
AutoRL, an evolutionary automation layer around reinforcement learning (RL) that searches for a deep RL reward and neural network architecture with large-scale hyper-parameter optimization.
It first finds a reward that maximizes task completion and then finds a neural network architecture that maximizes the cumulative of the found reward.

## Kwa: Balancing collective exploration and exploitation

The effectiveness of these strategies has been shown to be related to the so-called exploration–exploitation dilemma: i.e., the existence of a distinct balance between exploitative actions and exploratory ones while the system is operating.
Recent results point to the need for a dynamic exploration–exploitation balance to unlock high levels of flexibility, adaptivity, and swarm intelligence.