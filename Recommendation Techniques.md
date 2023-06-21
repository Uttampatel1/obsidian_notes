There are several types of product recommendation systems, each based on different [machine learning algorithms](https://www.netguru.com/blog/supervised-machine-learning) which are used to conduct the data filtering process. The main categories are content-based filtering (CBF), collaborative filtering (CF), complementary filtering, and hybrid recommendation systems, which use a combination of CBF and CF.

### Content-based filtering:

- CBF tracks a user’s actions, such as products bought or clicked on, web pages viewed, time spent browsing various product categories, etc. It then uses this information to create a customer profile. 
- This profile is then compared to the product catalogue to make recommendations.

### Collaborative filtering:

- CF methods involve collecting and analysing information on users’ behaviours and preferences, and predicting what each user will like based on their similarity to other users. 
- For example, on a music streaming site, if User A likes the bands Radiohead, R.E.M., and U2, and User B likes Radiohead, R.E.M., and Pearl Jam, then the CF filtering algorithm will determine that the two users have similar tastes, and will recommend Pearl Jam to User A, and U2 to User B. Similarities between pairs of items (or bands, movies, TV shows or anything else) can be determined in the same way. In this example, since both users like the bands Radiohead and R.E.M., the pairing would receive a positive similarity score. 
- The algorithms most frequently used in CF filtering are the _k_-nearest neighbours algorithm, and latent factor analysis (LFM).

### Complementary filtering:

- Here, the system learns the probability of two or more products being bought together. For example, when a user buys a smartphone from an ecommerce store, it is more probable that the same user will buy a set of headphones on a return visit, rather than another smartphone.
- As such, the algorithms are based around recommending products that are complementary to other products – they are product-defined, as opposed to user-defined, as in CBF and CF. 
- The Naïve Bayes algorithm is most commonly used in complementary filtering.

### Hybrid recommendation systems:

- Hybrid approaches essentially work by combining CBF and CF methods. This can be achieved in a number of ways – for example, by making content-based and collaborative-based predictions separately and then combining them, by adding collaborative-based capabilities to a content-based approach (and vice versa), or by purposefully unifying the two approaches into one model.

[[Recommendation System Algorithms]]
