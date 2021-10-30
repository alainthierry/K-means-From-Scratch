# `K-means-From-Scratch`

This implementation is done for the purpose of understanding and getting an overview of clustering and K-means especially. Any suggestions for improvement are welcome or something the same way is warmly welcome.


## `K-means : How it works ?`

`1` Select a number of classes/groups and randomly initialize their respective center
`2` Each data point is classified by computing the distance between that point and each closest center
`3` Based on these classified points, we recompute the group center by taking the mean of all the vectors in that group.
`4` The steps `2 and 3` are repeated for a certain number of iterations until the group center 
doesn't change between iterations. `The inertia helps assess the model.`

https://www.sciencedirect.com/science/article/pii/S1875389212006220", https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html">scikit-learn, https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68",
https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68

## `Implementation`
The first implementation is structured in functions, and the second one combines/groups those functions in a single class, in order to add specifically the `predict` method and to run some
unit test of the class methods.

## `Data sources`

The data used to assess the implementation is available on https://ciqual.anses.fr/.
And only five food constituents are used `energy, water, proteins, carbohydrates, and lipids`

## `Implementation comparison - Scikit-learn and This one`

The scikit-learn implementation wins only regarding the running time.
Considering other aspects, the both implementations perform identically.

## `Requirements.txt`

Use the requirements.txt file to have an overview of used libraries and their
versions to avoid some warnings and issues.
