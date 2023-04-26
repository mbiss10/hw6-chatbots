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

Yes 

- We attribute humnan level CAPABILITIES (ability to reason) and VALUES (fairness, bias, etc. -- reccomend horror movie to kid)
  to the chatbot and it has neither

- Like OpenAI, prefix with disclaimer "I am an AI..."
- Avoid answering certain things (e.g. harm classification)
- Build guardrails e.g. checking user's age before making reccs 

"""

######################################################################################

"""
QUESTION 2 - Privacy Leaks

One of the potential harms for chatbots is collecting and then subsequently leaking 
(advertly or inadvertently) private information. Does your chatbot have risk of doing so? 
Can you think of ways that designers of the chatbot can help to mitigate this risk? 
"""

Q2_your_answer = """

- We store movie preferences transiently but make sure to clear it out between interactions
- We don't collect user info so even leaks wouldn't be traced back to specific people 

- We don't fine tune or do extra training with user preferences 

- There is maybe info in the movie ratings? E.g. differential privacy -- if only one person has 
  ever seen a movie, then you know the rating for that movie came from that person 

- ENCRYPT
- store model on local devices (train in federated way)
- just don't store data long term
- have users opt in if you're training on their data 
- collect minimal data

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