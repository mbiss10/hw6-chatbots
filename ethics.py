"""
Please answer the following ethics and reflection questions. 

We are expecting at least three complete sentences for each question for full credit. 

Each question is worth 2.5 points. 
"""

######################################################################################
"""
QUESTION 1 - Anthropomorphizing

Is there potential for users of your chatbot possibly anthropomorphize 
(attribute human characteristics to an object) it? 
What are some possible ramifications of anthropomorphizing chatbot systems? 
Can you think of any ways that chatbot designers could ensure that users can easily 
distinguish the chatbot responses from those of a human?
"""

Q1_your_answer = """

Yes, there is definitely potential for users of our chatbot to anthropomorphize it. In fact, we 
made it a goal to make the chatbot seem friendly, personable, and even funny at times in order
to make it easier and more enjoyable to use. Some users could mistake this personality as a 
sign that the chatbot has additional underlying human-like characteristics, but part of the 
reason that anthropomorphization was not a concern for us on this project is that the chatbot
is still very rudimentary and we know it will only be used by those who are aware that it is a 
CS-375 homework assignment.

In general, we view the ramifications of anthropomorphizing chatbot systems as falling into two
categories: users attributing humnan-level CAPABILITIES to the chatbot, and users attributing 
human VALUES to the chatbot. Both can be dangerous. If users think the chatbot has a human-like 
ability to reason, they might overestimate the confidence they should have in its results. It's
important for users to know that AI systems are capable of making errors (even when they report
results confidently) which is no big deal for a movie reccomender but could be harmful if, for
example, you're asking it for medical advice. It's also potentially dangerous to assume that AI
systems are somehow imbued with human values such as fairness and compassion. Our movie bot would
have no qualms about reccomending R-rated horror movies to a 5-year-old if it determined 
these were the statistically most similar movies to the preferences it received, even if a human
would have the judgement to not do so. 

Chatbot designers could ensure that users can easily  distinguish the chatbot responses from
those of a human by using strategies such as:
- Prefixing responses with disclaimers, like how ChatGPT often answers questions by starting
  with "as an AI language model developed by OpenAI, I ..."
- Build UIs that try to avoid misleading users. For example, customer service chatbots should
  probably not include images of actual people just to give the impression that users are
  talking to a human.
- Train the chatbot to have a particular style of language that could be more easily interpretted
  as not being a real person (e.g. a more direct tone, without use of slang or mentions of emotions)
- Building guardrails around certain sensitive topics or phrases to make it even more explicit
  that the user is interacting with an AI when these are discussed(or avoid answering 
  questions about them altogether).

"""

######################################################################################

"""
QUESTION 2 - Privacy Leaks

One of the potential harms for chatbots is collecting and then subsequently leaking 
(advertly or inadvertently) private information. Does your chatbot have risk of doing so? 
Can you think of ways that designers of the chatbot can help to mitigate this risk? 
"""

Q2_your_answer = """

Our chatbot is fairly resilient to potential privacy leaks since it does not collect much
private data and it is relatively stateless. After each session runs on the command line, 
the entire chatbot's state is destroyed and there are no logs of the conversation stored
anywhere. It is hard to imagine potential privacy leaks since it runs locally on a user's
machine, rather than over any sort of network. We store no personally identifiable data
about users -- only the bear minimum of what is needed for our reccomender (i.e. only the
user's movie preferences, not anything else that they said in the converastion) and 
even this information is stored transiently. We also do not use users' preferences to
fine-tune our statistical model, writeback to a central database, or perform any extra ML
training. If anything, the main privacy risk for this chatbot is being able to identify
information about the reviewers from IMDB since this data is more extensive than what 
we store from our chatbot interactions, but this is data that's publicly available online
and it is the responsibility of reviewers themselves to avoid publishing reviews that could
harm their privacy. With all that said, there are additional privacy features that could
be incorporated into our chatbot and other similar systems such as:
- Differential privacy: using statistical methods to avoid sharing data about individuals 
  when this data could be used to specifically identify them.
- Encrypting data (both at rest and in transit if using a network)
- Ensuring any data that is collected is as minimal as possible and remains on the user's
  device whenever possible. Federated learning provides a way to train models on data
  that lives on client devices without collecting the data into a single server.
- Store data for as short a time as possible. For our chatbot, this could also mean
  implementing a timeout feature that ends a conversation after some number of inactive
  minutes to avoid someone inadvertently leaving it running while it has their preferences
  stored.
- Always inform users about how their data will be used so they can choose to opt out
  if they want, and give them the option to delete all their data (the "right to be
  forgotten" as the GDPR calls it).

"""

######################################################################################

"""
QUESTION 3 - Effects on Labor

Advances in dialogue systems, and the language technologies based on them, could lead to the automation of 
tasks that are currently done by paid human workers, such as responding to customer-service queries, 
translating documents or writing computer code. These could displace workers and lead to widespread unemployment. 
What do you think different stakeholders -- elected government officials, employees at technology companies, 
citizens -- should do in anticipation of these risks and/or in response to these real-world harms? 
"""

Q3_your_answer = """

- Retraining programs
- UBI -- if AI truly increases productivity expontentially, there should be plenty of resources to go around for everyone 
- Learn to understand the systems to avoid issues like in Q1. Broad education about these systems solves a lot of problems


"""

"""
QUESTION 4 - Refelection 

You just built a frame-based dialogue system using a combination of rule-based and machine learning 
approaches. Congratulations! What are the advantages and disadvantages of this paradigm 
compared to an end-to-end deep learning approach, e.g. ChatGPT? 
"""

Q4_your_answer = """

- Closed domain (movies only) 
- Complex to build -- and very little can be ported over to new domains
- Brittle rule based dialogue framework
- Worse performance (just not as generalizable)
- Required domain specific labeled data (as opposed to webscale free text)

"""