1. How many training series? What fraction actually have a break
2. What are the distribution sof len(x_hist) and len(x_online)
3. What's the distribution of tau? And of tau / len(x_online) -- is the break position uniform, or biased?
4. Verify the docs' claim: is mean(x_hist) = 0 and std(x_hist) = 1 for every series?
5. For every series with a break, compute the before-vs-after difference across the break in: mean, standard deviation, and lag-1 autocorrelation Plot the joint distributions. What fraction of breaks are mean shifts, what fraction are variance shifts, what fraction are dependence changes, and how big are they?
6. Plot six series with the break marked: three with breaks (pick a small, a medium, and a large one by whatever "size" measure you built in #5), three without. Can you see the break by eye? 


1. 4967 (49.7%)a
2. 1000-5000, 10-999
3. uniform, no usable tilt
4. 