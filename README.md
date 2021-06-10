# OnionOrNot
"Onion or Not" is a simple machine learning project which uses ML Methods and Algorithms to develop and save a Machine Learning Model that can accurately distinguish between  articles from the satire news site 'The Onion' and absurd but real news headlines. To be more specific, the ML Algorithm maps input headlines to a TFIDF Matrix(where each vector in the matrix represents a headline), and uses the Logistic Regression Algorithm to develop a linear seperation between vectors that represent the two types of headlines. This linear seperation can then be utilized to predict future vectorized input headlines as either from the Onion or as real. This method ended up with a testing accuracy of 85.2%.

The second part of this project involved using the ML model to develop a simple local server website to easily insert input text headlines, and get the prediction on whether it's from the Onion or not. This was made possible due to Flask and HTML. The ending result:

https://user-images.githubusercontent.com/77365987/119209575-30000f80-ba5c-11eb-900d-5658656d2068.mov

Dataset used for arising the ML model: https://github.com/lukefeilberg/onion
