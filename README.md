# OnionOrNot
"Onion or Not" is a simple machine learning project which uses ML Methods and Algorithms to develop and save a Machine Learning Model that can accurately distinguish between  articles from the satire news site 'The Onion' and absurd but real news headlines. To be more specific, the ML Algorithm maps input headlines to a TFIDF Matrix(where each vector in the matrix represents a headline), and uses the Logistic Regression Algorithm to develop a linear seperation between vectors that represent headlines from the Onion and vectors that represent real headlines. This linear seperation can then be utilized to predict future vectorized input headlines as either from the Onion or as real. This method ended up with a testing accuracy of 84.41%.

The second part of this project involved using the ML model to develop a simple local server website to easily insert input text headlines, and get the prediction on whether it's from the Onion or not. This was made possible due to Flask and HTML. The ending result 
https://user-images.githubusercontent.com/77365987/118757312-b5e44680-b821-11eb-8652-b6a08e4f3a93.mov

