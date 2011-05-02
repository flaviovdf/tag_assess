Experiments
===========

First sketch
------------

The main idea of the experiment is to create a social search similar to the work in [1]_. In this model,
users browse a tag graph (defined by co-occurrence of tags in items, users or both) in order to reach a 
topic of interest (another tag). When browsing a tag, a user has only the knowledge of neighbors of that tag. 
A good 'user intent model' will guide users to the topic of interest. 

We can change this idea slightly in order to perform our assessment of value. Firstly, the goal vertex will be an item annotated by some tag. On each step, the user will chose the tag with the most value to continue. We want to be as close to the shortest path as possible (which we know for the whole graph) from the initial tag to the goal.

::

    def experiment_1(annotations, user, user_vocabulary, relevant_items_for_user):
        trip_graph = build_tripartite_graph(annotations)
        tag_tag_graph = collapse(trip_graph, 'items')
    
        #We can collapse on items, users or both. We want this tag to be directed but
        #edges between to tag vertex will always have another on the other way. That is,
        #edge (t, t') exists if and only if edge (t', t) also exists. This will not be the 
        #same between tag and items.
                
        search_graph = tag_tag_graph
        for tag in trip_graph.tags():
            for item in trip_graph.get_items_connected_to(t):
                search_graph.add_edge((tag, item)) #Add a directed edge from tag to item
        
        results = defaultdict(list) #This is basically a map to a list.
        for item in sort_by_relevance(relevant_items_for_user):
            for j from 0 to min(100, len(user_vocabulary)):
                initial_query = user_vocabulary.rand_tag()
                initial_vertex = search_graph.vertex(initial_query)
                
                baseline = search_graph.distance(initial_query, i) #Can be pre-computed
                
                found = False
                steps = 1
                query = initial_query
                while not found:
                    neighbors = search_graph.neighbors(initial_query)
                    if item in neighbors:
                        found = True
                    else:
                        max_importance = 0
                        max_importance_tag = None
                        for neighbor in neighbors:
                            if neighbor.type() == 'tag':
                                importance = get_tag_importance(user, neighbor) #This is Elizeus method. (pre-computed) 
                                if importance > max_importance:
                                    max_importance = importance
                                    max_importance_tag = neighbor
                        
                        query = max_importance_tag
                        steps += 1
                
                results[item].append(steps)
                
        return results
                        
    
Things to consider:

1. There may not be a path
2. The neighbors list may be empty (or contain only items). Meaning that we got lost
3. Return to a tag which we previously considered. What to do? Error? Find new path?
4. Tags with the same importance (ties)
5. We may want to measure not only the amount of steps, but also the amount of garbage
   (irrelevant items) associated with the tag. A correct assessment of value should:
   
    * Minimize the effort (steps)
    * Minimize the garbage returned (uncertainty)
    * Maximize the average relevance

Questions
---------

1. How can we model AND search? The sketch above is based on the assumption that a user only
uses one term per query at a time. 

(Idea 1: Simple and easy)
Elizeu suggested we create super-vertexes, which correspond to more than one tag. Edges between these nodes and items exist only if the item is annotated by all edges.

(Idea 2: Complex)
There is this one interesting paper by Yahoo! Research which models query reformulation [2]_.
Basically, for a given query a user can:

1. Change some terms with the same intent (Correction, C)
2. Add terms (Specify, S)
3. Remove terms (Generalize, G)
4. Move away and focus on a new query (Parallel move, P)

The experiment above can be seen as modeling a user which always performs C operations. But we 
may do better, by simulating a user which can perform steps C, S or G until the item is found.
We would the measure the amount of changes (CSG) from a initial query which leads the user to the item
she wants. This may be interesting because we can model multi-term queries under the **and** model. 
The **or** model can also be modeled (or even complex queries), but we can leave this out because:

1. **AND** is more common on the web [3]_
2. **AND** is the default by Delicious, MrTaggy and Flickr
  
References
----------
    
.. [1] Pragmatic Evaluation of Folksonomies. Helic et al. WWW 2011
.. [2] Query reformulation mining: Models and patterns. Boldi et al. Web mining for search - Springer. Also Web intelligence 2010.
.. [3] Modern information retrieval. Yates and Berthier. 2nd edition (the chapter on indexing)