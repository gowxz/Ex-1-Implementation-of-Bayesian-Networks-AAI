<H3 ALIGN=RIGHT> DATE:<H3>

<H1 ALIGN=CENTER> Experiment-1: Implementation of Bayesian Networks</H1>

### Name: Gowtham S
### Register Number: 2305002008


## Aim:

To create a Bayesian Network for the given dataset in Python.

    
## Algorithm:

**Step 1:** Import necessary libraries: pandas, networkx, matplotlib.pyplot, Bbn, Edge, EdgeType, BbnNode, Variable, EvidenceBuilder, InferenceController.

**Step 2:** Set pandas options to display more columns.

**Step 3:** Read in weather data from a CSV file using pandas.

**Step 4:** Remove records where the target variable RainTomorrow has missing values.

**Step 5:** Fill in missing values in other columns with the column mean.

**Step 6:** Create bands for variables that will be used in the model (Humidity9amCat, Humidity3pmCat, and WindGustSpeedCat).

**Step 7:** Define a function to calculate probability distributions, which go into the Bayesian Belief Network (BBN).

**Step 8:** Create BbnNode objects for Humidity9amCat, Humidity3pmCat, WindGustSpeedCat, and RainTomorrow, using the probs() function to calculate their probabilities.

**Step 9:** Create a Bbn object and add the BbnNode objects to it, along with edges between the nodes.

**Step 10:** Convert the BBN to a join tree using the InferenceController.

**Step 11:** Set node positions for the graph.

**Step 12:** Set options for the graph appearance.

**Step 13:** Generate the graph using network

**Step 14:** Update margins and display the graph using matplotlib.pyplot.

---

## Program:

```python
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pybbn.graph.dag import Bbn
from pybbn.graph.dag import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController

# Display up to 50 columns in pandas DataFrame
pd.options.display.max_columns = 50

# Load the CSV file
df = pd.read_csv('weatherAUS.csv', encoding='utf-8')

# Filter out rows with null values in the 'RainTomorrow' column
df = df[pd.isnull(df['RainTomorrow']) == False]

# Fill NaN values only for numeric columns
df = df.fillna(df.mean(numeric_only=True))

# Apply categorical transformations
df['WindGustSpeedCat'] = df['WindGustSpeed'].apply(lambda x: '0.<=40' if x <= 40 else '1.40-50' if 40 < x <= 50 else '2.>50')
df['Humidity9amCat'] = df['Humidity9am'].apply(lambda x: '1.>60' if x > 60 else '0.<=60')
df['Humidity3pmCat'] = df['Humidity3pm'].apply(lambda x: '1.>60' if x > 60 else '0.<=60')

# Print the DataFrame to check if transformations are correct
print(df.head())

# Function to calculate probabilities
def probs(data, child, parent1=None, parent2=None):
    if parent1 is None:
        # Calculate probabilities for a node without parents
        prob = pd.crosstab(data[child], 'Empty', margins=False, normalize='columns').sort_index().to_numpy().reshape(-1).tolist()
    else:
        # Check if the child node has 1 parent or 2 parents
        if parent2 is None:
            # Calculate probabilities for a node with 1 parent
            prob = pd.crosstab(data[parent1], data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
        else:
            # Calculate probabilities for a node with 2 parents
            prob = pd.crosstab([data[parent1], data[parent2]], data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    return prob

# Define nodes with variables and probabilities
H9am = BbnNode(Variable(0, 'H9am', ['<=60', '>60']), probs(df, child='Humidity9amCat'))
H3pm = BbnNode(Variable(1, 'H3pm', ['<=60', '>60']), probs(df, child='Humidity3pmCat', parent1='Humidity9amCat'))
W = BbnNode(Variable(2, 'W', ['<=40', '40-50', '>50']), probs(df, child='WindGustSpeedCat'))
RT = BbnNode(Variable(3, 'RT', ['No', 'Yes']), probs(df, child='RainTomorrow', parent1='Humidity3pmCat', parent2='WindGustSpeedCat'))

# Create the Bayesian Belief Network
bbn = Bbn() \
    .add_node(H9am) \
    .add_node(H3pm) \
    .add_node(W) \
    .add_node(RT) \
    .add_edge(Edge(H9am, H3pm, EdgeType.DIRECTED)) \
    .add_edge(Edge(H3pm, RT, EdgeType.DIRECTED)) \
    .add_edge(Edge(W, RT, EdgeType.DIRECTED))

# Apply inference
join_tree = InferenceController.apply(bbn)

# Set positions for nodes in the graph
pos = {0: (-1, 2), 1: (-1, 0.5), 2: (1, 0.5), 3: (0, -1)}

# Define graph drawing options
options = {
    "font_size": 16,
    "node_size": 4000,
    "node_color": "white",
    "edgecolors": "black",
    "edge_color": "red",
    "linewidths": 5,
    "width": 5,
}

# Convert the BBN to a NetworkX graph and draw it
n, d = bbn.to_nx_graph()

fig, ax = plt.subplots(figsize=(6, 6))

nx.draw(n, pos=pos, with_labels=True, labels=d, ax=ax, **options)

ax.margins(0.10)
plt.axis("off")
plt.show()

```

---

## Output:



---

## Result:
   Thus, a Bayesian Network is generated using Python.
