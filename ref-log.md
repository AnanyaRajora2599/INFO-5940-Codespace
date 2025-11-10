# ref-log.md

## Reflection
My experience developing multi agent travel planner showed me how cooperation between different types of agents can yield the results that are not only more diverse but also more trustworthy than the ones produced by single model pipeline. The planner and reviewer acted as two allies, one was imaginative and other one was critical. I came to a realization that I could create prompts which give each agent separate and distinct boundaries. Planner had to develop itinerary that was complete and self contained using only its internal knowledge while reviewer was to act as fact checker using the real time internet searches. This division of labor made me realize the significance of role definition and memory isolation and also information flow in agent systems.

The main task was to manage prompt length and its clarity. When planner prompt was too imprecise it delivered fatty itineraries, when too stringent it restricted inventiveness. And same for reviewer as it had to be brief yet very critical not cutting off the case but just reporting findings. The other problem was how to get internet_search tool logging to work live in streamlit without disrupting async requests. I overcame this by lightweight logger that would capture and flash each event of the tool used in sidebar which also provide visual aid for debugging search calls.

I concentrated my imagination on the areas of prompt engineering and the way output was structured. Replies of the two agents were structured and presented in markdown with clear sections, trip overview, itinerary by day and delta list. In this way not only overall readability was enhanced but planner to reviewer pipeline also appeared to be a professional peer review process. Additionally I highlighted cost tracking and feasibility reasoning which contributed to itinerary being perceived as real and anchored.

## GPT Prompts
1. How to use multiple agents in an agentic pipeline using prompts  
2. How to stress on specific tokens in a Prompt for llm  
3. What is best way to give output format to llm
```

