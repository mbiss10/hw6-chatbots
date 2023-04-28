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
import random
from enum import Enum
import util


class Chatbot:
    """Class that implements the chatbot for HW 6."""

    def __init__(self):
        self.name = 'I-Am-Deebee' # other options: Query Tarantino, Martin (F-Score)sese

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, self.ratings = util.load_ratings('data/ratings.txt')
        
        # Load sentiment words 
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        # Train the classifier
        self.train_logreg_sentiment_classifier()

        # Possible choices for responses to a user's positive/negative movie preferences
        self.positive_responses = [
            ("Yeah, ", " is such a dope movie!"),
            ("Nice, I also like ", "."),
            ("Ooh good call, ", " is great!"),
        ]
        self.negative_responses = [
            ("Got it, so you did not like ", "."),
            ("Good to know that you didn't like ", "."),
            ("Right, makes sense that you didn't like ", "... that movie it terrible!"),
        ]

        self.TAGS_TO_POS_MAP = {'VERB':'v', 'NOUN':'n', 'PRON':'n' , 'ADJ':'a', 'ADV':'r'}

        # Bot state for each turn of conversation
        self.curr_processing_raw_title = None  # raw title is what the user entered
        self.curr_processing_idx = None  # single movie index for the user's current response
        self.curr_processing_indices = None  # all possible movie indices for the user's current response
        self.curr_processing_sentiment = None  # inferred sentiment for the user's current response
        self.is_disambiguating = False  # whether the bot is currently trying to disambiguate user's response

        # user preferences learned so far, as a dict mapping movie indices -> integer scores (+/-1)
        self.user_reviews = dict()

        self.use_lemmatizer = False

        # list of reccomendations being given out by the bot
        self.reccomendations = None

        self.articles = ["A", "An", "The"]
        self.titles_articles_handled = self.init_titles_articles_handled()

        nltk.data.path.append('./deps/nltk_data/')
        self.lemmatizer = nltk.stem.WordNetLemmatizer()


    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        return f"Hey there! Get ready to meet {self.name}, a robot that will help you find movies you'll love. \nTo exit, type \":quit\" (or press Ctrl-C to force the exit)\n"


    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""

        greeting_message = f"What's up, {self.name} here! Let's find you a great movie. First, I need to get a sense for your taste. Before we begin, we have an expiremental lemmatizer that could improve our performance. If you would like to use it, respond 'yes' at any time. If you would like to disable it, respond 'no' at any time. You can also leave it disabled by telling us about a movie you have seen."
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """

        goodbye_message = "Oh, guess we're done here. Hope you found a movie you'll like. Cya! 👋"
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

    # Static method. Returns year from a movie title, or None if no year is found.
    # Assumes year is in parentheses at end of string.
    def get_year_from_title(title: str) -> str:
        res = re.match(r'.* \((\d{4})\)$', title)
        return res[1] if res is not None else None

    def get_response(self, sentiment, title, is_fifth=False):
        if sentiment.lower() == "negative":
            prefix, suffix = random.choice(self.negative_responses)
        else: 
            prefix, suffix = random.choice(self.positive_responses)

        res = prefix + title + suffix

        if is_fifth:
            res += "\nI've sucked up enough of your data to train my internal neural network superintelligence transformer recurrent convolutional system. Are you ready to hear my reccomendation? (If not, answer ':quit' to end our conversation.)"

        return res

    def reset_state(self):
        """
        Resets all of the bot's state variables for a new turn of the conversation.
        """
        self.curr_processing_raw_title = None
        self.curr_processing_indices = None
        self.curr_processing_sentiment = None
        self.curr_processing_idx = None
        self.is_disambiguating = False

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

        response = ""

        # if the user enters 'yes' and we are not reccomending, then we toggle the lemmatizer on
        if line.lower().strip() == 'yes' and len(self.user_reviews) < 5:
            self.use_lemmatizer = True
            return "Okay! We will use the lemmatizer. Tell us about a movie you have seen."
        
        # if the user enters 'no' and we are reccomending, then we toggle the lemmatizer off
        elif line.lower().strip() == 'no' and len(self.user_reviews) < 5:
            self.use_lemmatizer = True
            return "Okay! We will won't use the lemmatizer. Tell us about a movie you have seen."

        if len(self.user_reviews) >= 5:
            if self.reccomendations is None:
                # get reccomendations for the first time
                self.reccomendations = self.recommend_movies(self.user_reviews, num_return=10)
            
            if len(self.reccomendations) == 0:
                # we've given out all recs
                return "That's all the reccomendations I've got for now! You should quit now using :quit."

            return f"I reccomend... {self.reccomendations.pop(0)}. \n Do you want another reccomendation? Type anything to receive another, or end our conversation by typing :quit."

        if self.curr_processing_raw_title is None:
            # extract titles
            titles = self.extract_titles(line)
            
            # did not find any titles... complain
            if not titles:
                return "I'm sorry, did you mention a movie in your last message? Please include any movie titles in quotes."

            # handle multiple titles. For now, just use the first one
            if len(titles) > 1:
                response += f"Thanks for the info! Let's just focus on one movie at a time. I'll start with {titles[0]}.\n"
            
            title = titles[0]
            self.curr_processing_raw_title = titles[0]

        if self.curr_processing_indices is None:
            # get indicies for movies
            indices = self.find_movies_idx_by_title(title)
            exact_titles = [self.titles[idx][0] for idx in indices]
            self.curr_processing_indices = indices
        
            # Chatbot does not know about the movie... complain
            if not indices:
                self.reset_state()
                return f"I'm sorry, I don't recognize the movie '{title}'."
            
            if len(self.curr_processing_indices) == 1 and self.curr_processing_indices[0] in self.user_reviews:
                self.reset_state()
                return "You've already told me about that movie!"

        if self.curr_processing_sentiment is None:
            # Get/clarify sentiment
            processed_line = self.lemmatize_and_mask_title(line) if self.use_lemmatizer else line
            predicted_sentiment_statistical = self.predict_sentiment_statistical(processed_line)
            predicted_sentiment_rule_based = self.predict_sentiment_rule_based(processed_line)
            chosen_sentiment_score = predicted_sentiment_rule_based

            self.curr_processing_sentiment = chosen_sentiment_score
            
            if chosen_sentiment_score == -1 and len(self.curr_processing_indices) == 1: 
                self.reset_state()
                self.user_reviews[indices[0]] = -1
                response += self.get_response("negative", title, is_fifth = len(self.user_reviews) >= 5)
                return response
                
            elif chosen_sentiment_score == 1 and len(self.curr_processing_indices) == 1: 
                self.reset_state()
                self.user_reviews[indices[0]] = 1
                response += self.get_response("positive", title, is_fifth = len(self.user_reviews) >= 5)
                return response

            elif chosen_sentiment_score == 0:
                # Important: we still don't know if we've disambiguated titles
                response += f"I'm sorry, I'm not quite sure if you liked '{title}'. Tell me more about what you thought about it."
                return response

        if self.curr_processing_sentiment == 0: 
            
            processed_line = self.lemmatize_and_mask_title(line) if self.use_lemmatizer else line
            predicted_sentiment_statistical = self.predict_sentiment_statistical(processed_line)
            predicted_sentiment_rule_based = self.predict_sentiment_rule_based(processed_line)
            chosen_sentiment_score = predicted_sentiment_rule_based
        
            self.curr_processing_sentiment = chosen_sentiment_score
            
            if chosen_sentiment_score == -1 and len(self.curr_processing_indices) == 1: 
                self.user_reviews[self.curr_processing_indices[0]] = -1
                response += self.get_response("negative", self.curr_processing_raw_title, is_fifth = len(self.user_reviews) >= 5)
                self.reset_state()
                return response
                
            elif chosen_sentiment_score == 1 and len(self.curr_processing_indices) == 1: 
                self.user_reviews[self.curr_processing_indices[0]] = 1
                response += self.get_response("positive", self.curr_processing_raw_title, is_fifth = len(self.user_reviews) >= 5)
                self.reset_state()
                return response

            elif chosen_sentiment_score == 0: # sentiment = 0
                response += f"I'm sorry, I'm still not quite sure if you liked '{self.curr_processing_raw_title}'. Tell me more about what you thought about it."
                return response
            

        if self.curr_processing_idx is None:
            if len(self.curr_processing_indices) == 1:
                if self.curr_processing_indices[0] in self.user_reviews:
                    self.reset_state()
                    return "You've already told me about that movie!"

                self.is_disambiguating = False
                self.curr_processing_idx = self.curr_processing_indices[0]

            elif self.is_disambiguating:
                # keep disambiguating
                disambiguation = self.disambiguate_candidates(line, self.curr_processing_indices)

                if len(self.curr_processing_indices) == len(disambiguation) or len(disambiguation) == 0:
                    # Either the user did not narrow down the set of candidates at all, or disambiguation failed.
                    self.reset_state()
                    return "Hmm, I'm having trouble understanding that response. Let's try this again from the top. Tell me about a movie you've seen, and try to be specific about the title!"

                elif len(disambiguation) == 1:    
                    self.curr_processing_idx = disambiguation[0]
                    self.is_disambiguating = False
                
                else:
                    exact_titles = [self.titles[idx][0] for idx in self.curr_processing_indices] # TODO: clean this up
                    return f"I still don't know which movie you meant: {' or '.join(exact_titles)}?\n"

            elif not self.is_disambiguating:
                # start disambiguating
                self.is_disambiguating = True
                exact_titles = [self.titles[idx][0] for idx in self.curr_processing_indices] # TODO: clean this up
                return f"I understand that you {'liked' if self.curr_processing_sentiment == 1 else 'disliked'} {self.curr_processing_raw_title}. There are multiple movies with that name. Which did you mean: {' or '.join(exact_titles)}?\n"

        if self.curr_processing_idx in self.user_reviews:
            self.reset_state()
            return "You've already told me about that movie!"

        if self.curr_processing_sentiment == -1: 
            self.user_reviews[self.curr_processing_idx] = -1
            response += self.get_response("negative", self.titles[self.curr_processing_idx][0], is_fifth = len(self.user_reviews) >= 5)
            self.reset_state()
            return response
                
        elif self.curr_processing_sentiment == 1: 
            self.user_reviews[self.curr_processing_idx] = 1
            response += self.get_response("positive", self.titles[self.curr_processing_idx][0], is_fifth = len(self.user_reviews) >= 5)
            self.reset_state()
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
        return list(re.findall(r'[\"](.*?)[\"]', user_input))

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
        res = set()

        pattern = title

        if Chatbot.get_year_from_title(title) is None:
            pattern += r'.* \(\d{4}\)$'    
        else: 
            pattern = re.escape(pattern)

        pattern_obj = re.compile(pattern, flags=re.IGNORECASE)

        for (idx, (candidate, _)) in enumerate(self.titles):
            if pattern_obj.search(candidate) is not None:
                res.add(idx)
        
        for (idx, candidate) in self.titles_articles_handled:
            if pattern_obj.search(candidate) is not None:
                res.add(idx)

        return list(res)


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
        
        candidate_titles = [(self.titles[idx][0], idx) for idx in candidates]
        filtered_candidates = [idx for title, idx in candidate_titles if re.search(re.escape(clarification), title, flags=re.IGNORECASE) is not None]
        return filtered_candidates

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
        user_input = user_input.lower()

        tokens = nltk.regexp_tokenize(user_input, r"\w+")

        sentiments = Counter([self.sentiment[token] for token in tokens if token in self.sentiment])
        score = sentiments["pos"] - sentiments["neg"]

        return 1 if score > 0 else (-1 if score < 0 else 0)
        

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

        vectorizer = CountVectorizer(min_df=20, #only look at words that occur in at least 20 docs
                                    stop_words='english', # remove english stop words
                                    max_features=1000, #only select the top 1000 features 
                                    )
        
        texts = [s.lower() for s in texts]
        X = vectorizer.fit_transform(texts)

        Y = np.array([1 if label.lower() == "fresh" else -1 for label in y])
    
        logistic_regression_classifier = sklearn.linear_model.LogisticRegression(penalty=None,  max_iter=500)
        logistic_regression_classifier.fit(X, Y, )

        # print("Model acc: ", logistic_regression_classifier.score(X, Y))

        self.model = logistic_regression_classifier #variable name that will eventually be the sklearn Logistic Regression classifier you train 
        self.count_vectorizer = vectorizer #variable name will eventually be the CountVectorizer from sklearn 


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
        x = self.count_vectorizer.transform([user_input])

        if np.sum(x) == 0:
            return 0
        
        return self.model.predict(x)[0]


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
        num_movies = self.ratings.shape[0]
    
        user_rating_all_movies = np.zeros(num_movies)
        for movie_idx, user_review in user_ratings.items():
            user_rating_all_movies[movie_idx] = user_review

        recs = util.recommend(user_rating_all_movies, self.ratings, num_return = num_return)
        return [self.titles[rec_idx][0] for rec_idx in recs]

    ############################################################################
    # 5. Open-ended                                                            #
    ############################################################################

    def init_titles_articles_handled(self):
        """
        Code for option 5 of the open-ended function1 part of this assignment for
        dealing with articles in movie titles. This method is called during the
        chatbot's initialization. It parses all titles in the movies.txt file
        and returns a list of tuples of the form (idx, processed_title) for 
        all movies that have articles in their title. The processed title has
        the article moved to the front of the title, to create a more natural
        name. 

        The articles that are supported are "A", "An", and "The". Note: one
        limitation of this that we are aware of is that some titles have non-
        English titles that also include articles, but we are not handling
        these cases right now. We checked all possible articles that appear
        at the end of a title following a comma and included these in the 
        file data/parsed_article_candidates.txt. This allowed us to confirm
        that A, An, and The should be sufficient for all English titles.
        
        For example:
        "American in Paris, An" => "An American in Paris"
        "Bothersome Man, The (Brysomme mannen, Den)" => "The Bothersome Man (Brysomme mannen, Den)"


        When trying to extract movie IDs that match a user's text entry, we 
        can iterate through both title lists to see if there are any matches
        in either format. In other words, a user is allowed to type either
        "American in Paris, An" OR "An American in Paris" and we'll know what
        they're talking about.
        """
        # Limitation: doesn't support non-English articles. This is a pattern with extra
        # articles we would support if we had more time.
        # pattern = r"(.*), (A|An|Un|The|Der|L'|Une|Les|Los|Il|Las|Det|Le|Das|El|De|En|Lo|Den|La)( \(|\))"

        pattern = rf"(.*), ({'|'.join(self.articles)}) (\(.*)"
        pattern_obj = re.compile(pattern, flags=re.IGNORECASE)
    
        res = []
        for idx, movie in enumerate(self.titles):
            title = movie[0]
            instances = pattern_obj.findall(title)
            if instances:
                first = instances[0]
                res.append((idx, first[1] + " " + first[0] + " " + first[2]))
        
        return res


    def lemmatize_and_mask_title(self, line : str) -> str:
        """
        The second extension function. It takes as input the user line
        and performs two prepocessing steps. First, it masks out any 
        movie titles in the input string. Then, it converts each token 
        in the string into its lemma. This helps with sentiment analysis.

        For example, given the input string 'I disliked "Good Will Hunting" 
        and hated "Lord of the Rings"', return 'I dislike I hate'.
        """
        # First, we mask out any title from the input string
        non_title_chars = []
        i = 0
        
        while i < len(line):
            if line[i] == "\"":
                # We encountered a quote and thus a movie title
                i += 1
                # Iterate unitl we find the corresponding close quote
                while i < len(line) and line[i] != "\"":
                    i += 1

            else:
                non_title_chars.append(line[i])

            i += 1

        line_no_titles = "".join(non_title_chars)

        #Split the masked string on whitespace
        tokens = nltk.regexp_tokenize(line_no_titles, r"\w+")

        #Tag each token with its part of speech
        tagged_tokens = nltk.tag.pos_tag(tokens, tagset='universal', lang='eng')

        # Use the tagged part of speech and the lemmatizer to lemmatize a given token
        lemmas = [self.lemmatizer.lemmatize(token, pos = (self.TAGS_TO_POS_MAP[pos] if pos in self.TAGS_TO_POS_MAP else 'n')) for token, pos in tagged_tokens]

        return " ".join(lemmas)

        # Now, we split as we do in the get_sentiment methods


    def function3(): 
        """
        Any additional functions beyond two count towards extra credit  
        """
        pass 


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')



