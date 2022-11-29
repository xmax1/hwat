
Edge comments between

```mermaid 
flowchart LR
	subgraph local 
		L
	end
    L(( fa:fa-circle-notch )) <--> Py
	subgraph  pyfig
		Py
	end
  subgraph cluster
		Clus
	end 
  subgraph cloud
		Cl
	end
  Py <--> Cl[fa:fa-cloud]
  Py[fa:fa-brain] <--> Clus[fa:fa-network-wired]
  Clus <--> Cl
```
