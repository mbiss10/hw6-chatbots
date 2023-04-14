import numpy as np
import argparse
import joblib
import re  
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
import nltk 
from collections import defaultdict, Counter
from typing import List, Dict, Union, Tuple

import util

class Chatbot:
    """Class that implements the chatbot for HW 6."""

    def __init__(self):
        # The chatbot's default name is `moviebot`.
        self.name = 'moviebot' # TODO: Give your chatbot a new name.

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, self.ratings = util.load_ratings('data/ratings.txt')
        
        # Load sentiment words 
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        # Train the classifier
        self.train_logreg_sentiment_classifier()

        # TODO: put any other class variables you need here 

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        return "Hi! I'm [NAME NEEDED]! I help you find movies that you will like. \n To exit: write \":quit\" (or press Ctrl-C to force the exit)"


    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""

        greeting_message = "Hi! I'm [NAME NEEDED]! I'm going to recommend a movie to you. First I will ask you about your taste in movies. Tell me about a movie that you have seen."
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """

        goodbye_message = "Have a nice day! 👋"
        return goodbye_message

    def debug(self, line):
        """
        Returns debug information as a string for the line string from the REPL

        No need to modify this function. 
        """
        return str(line)

    ############################################################################
    # 2. Extracting and transforming                                           #
    ############################################################################

    # Returns year from a movie title, or None if no year is found
    # Assumes year is in parentheses at end of string
    def get_year_from_title(self, title: str) -> str:
        res = re.match(r'.* \((\d{4})\)$', title)
        return res[1] if res is not None else None

    def process(self, line: str) -> str:
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this script.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'
        
        Arguments: 
            - line (str): a user-supplied line of text
        
        Returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################

        # these titles are raw strings
        titles = self.extract_titles(line)
        if not titles:
            return "I'm sorry, was there a movie in your statement? Please include any movie titles in quotes."
    
        if len(titles) > 1:
              print(f"Thanks for the info. Let's just focus on one movie at a time. I'll start with {titles[0]}")
                
        title = titles[0]

        indices = self.find_movies_idx_by_title(title)
        if not indices:
            # TODO: what if first of N movies is unrecognized?
            return "I'm sorry, I don't recognize the movie '{title}'"
    
        while len(indices) != 1:
            # disgambiguate
            exact_titles = [self.titles[idx][0] for idx in indices]
            user_clarification = input(f"Which movie did you mean: {' or '.join(exact_titles)}?\n")

            disambiguation = self.disambiguate_candidates(user_clarification, indices)

            if len(disambiguation) > 0:
                # If the disambiguation fails, it will return an empty list. We just retry with the full set of indices
                indices = disambiguation
        
        movie_idx = indices[0]
        exact_title = self.titles[movie_idx][0]

        # predict sentiment
        predicted_sentiment_rule_based = 1 # self.predict_sentiment_rule_based()
        predicted_sentiment_statistical = 1 # self.predict_sentiment_statistical()

        if predicted_sentiment_statistical == 0:
            response = f"I'm sorry, I'm not quite sure if you liked '{movie_idx}'. Tell me more about what you thought about it."
        elif predicted_sentiment_statistical == -1:
            response = f"Ah, so you didn't like '{movie_idx}'. Yeah, that movie sucks. Tell me about another movie you have seen."
        elif predicted_sentiment_statistical == 1:
            response = f"Yeah, '{movie_idx}' is a dope movie. Tell me about another movie you have seen."
        else:
            print("AHHH what is that sentiment?!")
            exit(1)

        return response

    def extract_titles(self, user_input: str) -> list:
        """Extract potential movie titles from the user input.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example 1:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I do not like any movies'))
          print(potential_titles) // prints []

        Example 2:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        Example 3: 
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'There are "Two" different "Movies" here'))
          print(potential_titles) // prints ["Two", "Movies"]                              
    
        Arguments:     
            - user_input (str) : a user-supplied line of text

        Returns: 
            - (list) movie titles that are potentially in the text

        Hints: 
            - What regular expressions would be helpful here? 
        """
        return list(re.findall(r'[\'"](.*?)[\'"]', user_input))

    def find_movies_idx_by_title(self, title:str) -> list:
        """ Given a movie title, return a list of indices of matching movies
        The indices correspond to those in data/movies.txt.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list that contains the index of that matching movie.

        Example 1:
          ids = chatbot.find_movies_idx_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        Example 2:
          ids = chatbot.find_movies_idx_by_title('Twelve Monkeys')
          print(ids) // prints [31]

        Arguments:
            - title (str): the movie title 

        Returns: 
            - a list of indices of matching movies

        Hints: 
            - You should use self.titles somewhere in this function.
              It might be helpful to explore self.titles in scratch.ipynb
            - You might find one or more of the following helpful: 
              re.search, re.findall, re.match, re.escape, re.compile
            - Our solution only takes about 7 lines. If you're using much more than that try to think 
              of a more concise approach 
        """
        res = []
        
        if title.lower()[:4] == "the ":
            title_no_the = title[4:]
            query = rf"(The )?{title_no_the}(, The)?"
        else:
            query = rf"{title}"

        if self.get_year_from_title(title) is None:
            query += r'.* \(\d{4}\)$'    
        else: 
            query = re.escape(query)
        
        # print("QUERY:")
        # print(query)

        for (idx, (candidate, _)) in enumerate(self.titles):
            if re.search(query, candidate, flags=re.IGNORECASE) is not None:
                res.append(idx)

        return res


    def disambiguate_candidates(self, clarification:str, candidates:list) -> list: 
        """Given a list of candidate movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (e.g. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)


        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If the clarification does not uniquely identify one of the movies, this 
        should return multiple elements in the list which the clarification could 
        be referring to. 

        Example 1 :
          chatbot.disambiguate_candidates("1997", [1359, 2716]) // should return [1359]

          Used in the middle of this sample dialogue 
              moviebot> 'Tell me one movie you liked.'
              user> '"Titanic"''
              moviebot> 'Which movie did you mean:  "Titanic (1997)" or "Titanic (1953)"?'
              user> "1997"
              movieboth> 'Ok. You meant "Titanic (1997)"'

        Example 2 :
          chatbot.disambiguate_candidates("1994", [274, 275, 276]) // should return [274, 276]

          Used in the middle of this sample dialogue
              moviebot> 'Tell me one movie you liked.'
              user> '"Three Colors"''
              moviebot> 'Which movie did you mean:  "Three Colors: Red (Trois couleurs: Rouge) (1994)"
                 or "Three Colors: Blue (Trois couleurs: Bleu) (1993)" 
                 or "Three Colors: White (Trzy kolory: Bialy) (1994)"?'
              user> "1994"
              moviebot> 'I'm sorry, I still don't understand.
                            Did you mean "Three Colors: Red (Trois couleurs: Rouge) (1994)" or
                            "Three Colors: White (Trzy kolory: Bialy) (1994)" '
    
        Arguments: 
            - clarification (str): user input intended to disambiguate between the given movies
            - candidates (list) : a list of movie indices

        Returns: 
            - a list of indices corresponding to the movies identified by the clarification

        Hints: 
            - You should use self.titles somewhere in this function 
            - You might find one or more of the following helpful: 
              re.search, re.findall, re.match, re.escape, re.compile
        """
        candidate_titles = [self.titles[idx][0] for idx in candidates]

        # If user clarifies using an exact match
        for idx, candidate_title in enumerate(candidate_titles):
            if candidate_title in clarification:
                return [candidates[idx]]

        # If user only specifies a year
        clarified_year = re.match(r'(\d{4})', clarification)
        if clarified_year is None:
            print("I'm sorry, I still don't understand.", end=' ')
            return []

        # Get year from the match object
        clarified_year = clarified_year[1]

        candidate_years = [self.get_year_from_title(candidate) for candidate in candidate_titles]
        return [candidates[idx] for idx, year in enumerate(candidate_years) if year == clarified_year]

    ############################################################################
    # 3. Sentiment                                                             #
    ########################################################################### 

    def predict_sentiment_rule_based(self, user_input: str) -> int:
        """Predict the sentiment class given a user_input

        In this function you will use a simple rule-based approach to 
        predict sentiment. 

        Use the sentiment words from data/sentiment.txt which we have already loaded for you in self.sentiment. 
        Then count the number of tokens that are in the positive sentiment category (pos_tok_count) 
        and negative sentiment category (neg_tok_count)

        This function should return 
        -1 (negative sentiment): if neg_tok_count > pos_tok_count
        0 (neural): if neg_tok_count is equal to pos_tok_count
        +1 (postive sentiment): if neg_tok_count < pos_tok_count

        Example:
          sentiment = chatbot.predict_sentiment_rule_based('I LOVE "The Titanic"'))
          print(sentiment) // prints 1
        
        Arguments: 
            - user_input (str) : a user-supplied line of text
        Returns: 
            - (int) a numerical value (-1, 0 or 1) for the sentiment of the text

        Hints: 
            - Take a look at self.sentiment (e.g. in scratch.ipynb)
            - Remember we want the count of *tokens* not *types*
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                                  
        return 0 # TODO: delete and replace this line
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def train_logreg_sentiment_classifier(self):
        """
        Trains a bag-of-words Logistic Regression classifier on the Rotten Tomatoes dataset 

        You'll have to transform the class labels (y) such that: 
            -1 inputed into sklearn corresponds to "rotten" in the dataset 
            +1 inputed into sklearn correspond to "fresh" in the dataset 
        
        To run call on the command line: 
            python3 chatbot.py --train_logreg_sentiment

        Hints: 
            - Review how we used CountVectorizer from sklearn in this code
                https://github.com/cs375williams/hw3-logistic-regression/blob/main/util.py#L193
            - You'll want to lowercase the texts
            - Review how you used sklearn to train a logistic regression classifier for HW 5.
            - Our solution uses less than about 10 lines of code. Your solution might be a bit too complicated.
            - We achieve greater than accuracy 0.7 on the training dataset. 
        """ 
        #load training data  
        texts, y = util.load_rotten_tomatoes_dataset()

        self.model = None #variable name that will eventually be the sklearn Logistic Regression classifier you train 
        self.count_vectorizer = None #variable name will eventually be the CountVectorizer from sklearn 

        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                                
        
        pass # TODO: delete and replace this line

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    def predict_sentiment_statistical(self, user_input: str) -> int: 
        """ Uses a trained bag-of-words Logistic Regression classifier to classifier the sentiment

        In this function you'll also uses sklearn's CountVectorizer that has been 
        fit on the training data to get bag-of-words representation.

        Example 1:
            sentiment = chatbot.predict_sentiment_statistical('This is great!')
            print(sentiment) // prints 1 

        Example 2:
            sentiment = chatbot.predict_sentiment_statistical('This movie is the worst')
            print(sentiment) // prints -1

        Example 3:
            sentiment = chatbot.predict_sentiment_statistical('blah')
            print(sentiment) // prints 0

        Arguments: 
            - user_input (str) : a user-supplied line of text
        Returns: int 
            -1 if the trained classifier predicts -1 
            1 if the trained classifier predicts 1 
            0 if the input has no words in the vocabulary of CountVectorizer (a row of 0's)

        Hints: 
            - Be sure to lower-case the user input 
            - Don't forget about a case for the 0 class! 
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                             
        return 0 # TODO: delete and replace this line
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    ############################################################################
    # 4. Movie Recommendation                                                  #
    ############################################################################

    def recommend_movies(self, user_ratings: dict, num_return: int = 3) -> List[str]:
        """
        This function takes user_ratings and returns a list of strings of the 
        recommended movie titles. 

        Be sure to call util.recommend() which has implemented collaborative 
        filtering for you. Collaborative filtering takes ratings from other users
        and makes a recommendation based on the small number of movies the current user has rated.  

        This function must have at least 5 ratings to make a recommendation. 

        Arguments: 
            - user_ratings (dict): 
                - keys are indices of movies 
                  (corresponding to rows in both data/movies.txt and data/ratings.txt) 
                - values are 1, 0, and -1 corresponding to positive, neutral, and 
                  negative sentiment respectively
            - num_return (optional, int): The number of movies to recommend

        Example: 
            bot_recommends = chatbot.recommend_movie({100: 1, 202: -1, 303: 1, 404:1, 505: 1})
            print(bot_recommends) // prints ['Trick or Treat (1986)', 'Dunston Checks In (1996)', 
            'Problem Child (1990)']

        Hints: 
            - You should be using self.ratings somewhere in this function 
            - It may be helpful to play around with util.recommend() in scratch.ipynb
            to make sure you know what this function is doing. 
        """ 
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                                    
        return [""]  # TODO: delete and replace this line
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    ############################################################################
    # 5. Open-ended                                                            #
    ############################################################################

    def function1():
        """
        TODO: delete and replace with your function.
        Be sure to put an adequate description in this docstring.  
        """
        pass

    def function2():
        """
        TODO: delete and replace with your function.
        Be sure to put an adequate description in this docstring.  
        """
        pass  

    def function3(): 
        """
        Any additional functions beyond two count towards extra credit  
        """
        pass 


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')



