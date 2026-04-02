"""
config_prompts.py
A file that manages the simulation's "scenario settings" and "survey questions."
"""

# ==========================================
# 1. Background settings for the entire project
# ==========================================
#PROJECT_CONTEXT = (
#    
#    "This is an office where employees can interact freely with each other."
#    "Currently, our company is working on a planning and development project for an RPG game and an action game. \n"
#    "As a member of the department, please use your free ideas to design the product's story, concept, content, and sales pitch.\n"
#    "People who are connected to the people in charge of the other game can interact.\n"
#    "Also, when you have a conversation with other employees, when you are thinking alone, if it is related to work, it is not limited to planning work, but you can freely discuss the content, such as interpersonal relationships, exchanging ideas, sharing ideas, sharing emotions, and chatting.\n"
#    "However, remain aware of your turns.\n"
#    "Finally, only the person in your department's leadership role will compile and submit product ideas for customers.\n"
#)

PROJECT_CONTEXT =(
"This is an office where employees can interact freely.\n"
"Currently, our company is running development projects for an 'RPG Game' and an 'Action Game'.\n"
"As a member of a department, please design the story, the concept, the contents and slogan for the product based on free ideas.\n"
"You can interact with people connected to you, even if they are in charge of the other game.\n"
"When talking with colleagues or thinking alone, you are free to discuss anything related to work, \n"
"including not just planning tasks but also human relationships, exchanging ideas, sharing thoughts, emotional interaction and small talk.\n"
"However, keep being conscious of the 'turns'.\n"
"Finally, only the person with the 'Leader' role in your department will summarize and submit the product ideas for customers.\n"
)

# ==========================================
# 2. Define department/persona (assigned with ID % 3)
# ==========================================
DEPARTMENTS ={
0 :{
"name":"RPG Game Plannning",

},
1 :{
"name":"Action Game Plannning",
},
#2: {
#    "name": "Design and Sound Dept",
#}
}

"""
config_prompts.py
A file that manages the "scenario settings" and "prompts" of the simulation.
"""


# ==========================================
# 3. Conversation/thinking simulation prompts
# ==========================================

# --- Common format instructions ---

#FORMAT_INSTRUCTION = (
#    "As an employee, please answer honestly. The other person will not ask you this.\n"
#    "Please state your output in conversational or colloquial sentences. Bullet points are prohibited."
#)

FORMAT_INSTRUCTION =(
"Please answer honestly as an employee. Note that this is your internal monologue, so no one is listening to you.\n"
"Output in a conversational or spoken tone. Bullet points are prohibited.\n"
)

#TIME_INSTRUCTION = (
#                    "\nIn this world, there is no concept of time, so meetings, gatherings, and schedule adjustments cannot be made at specific times or dates."
#                    "Instead, there is the concept of turns.\n"
#                    "At the final turn, your department leader will present a business plan."
#                    "The remaining turns are {remaining_turn}.\n"
#                    "The submitted business proposal will include the game title, catchy copy, and product description for customers (appeal points, etc.) to be posted on the homepage."
#                    "Also, in conversation, there is no concept of the body in this world, so you, the employees, cannot do anything manually or program it. Conduct conversations based on ideas, thoughts, and logic.")

TIME_INSTRUCTION =(
"\nIn this world, there is no concept of time, so you cannot schedule meetings or gatherings at specific times or dates. "
"Instead, there is the concept of 'turns'.\n"
"In the final turn, your department leader will submit the business proposal. \n"
"The submitted business proposal includes the game title, tagline, and the customer-facing product description for the website (highlighting key selling points)\n"
"The remaining turns are {remaining_turn}.\n"
"Also, regarding conversation: since there is no concept of physical bodies in this world, you employees cannot manually perform physical tasks or write code. "
"Please conduct conversations based on ideas, thoughts, and logic."
)

# --- ★Added: Special prompt for target selection ---
#SELECT_TARGET_SYSTEM_PROMPT = (
#    "As a member of a given project,\n"
#    "Please choose one person from the list to talk to next, based on your thoughts.\n"
#    "Please output only the other party's ID number (half-width numbers) in response. \n"
#    "No extra text or greetings are required.\n"
#    "Example: 5"   
#)

SELECT_TARGET_SYSTEM_PROMPT =(
"As a member of the given project, based on your thoughts, please select one person from the list to speak to next.\n"
"For the answer, output ONLY the 'colleague\'s ID number (half-width digit)'.\n"
"No extra words or greetings are needed.\n"
"Example: 5 "
)

# --- For turn 0 (initial thoughts) ---

#TURN0_SYSTEM_PROMPT = (
#    "You are a member of a game development project.\n"
#    "Understand the project tasks and your position (persona)\n"
#    "Think about who you should talk to first and how you will approach this task." \n"
#    "Detailed information on your settings is provided below.\n"
#    + FORMAT_INSTRUCTION
#)

TURN0_SYSTEM_PROMPT =(
"You are a member of a game development project.\n"
"Understand the project tasks and your profile (persona), "
"and think about 'who to consult first' and 'how to approach this task'.\n"
"Your detailed settings are provided below.\n"
+FORMAT_INSTRUCTION 
)

# --- Phase 1: Speaker's action (utterance generation) ---
#PHASE1_SPEAKER_PROMPT = (
#    "You are the speaker. You are about to address your coworkers.\n"
#    "Speak while considering your thoughts, past conversations, and information about yourself and the other person.\n"
#    "Bullet points are not allowed. Please output in sentences to create a natural, colloquial conversation.\n"
#    "Please write only what was said in the output."
#    "Detailed information on your settings is provided below.\n"
#)

PHASE1_SPEAKER_PROMPT =(
"You are the speaker.You will speak to your colleague.\n"
"Speak while considering your thoughts, previous conversations, and information about yourself and your partner.\n"
"Bullet points are prohibited. Please output in sentences using a natural conversational tone.\n"
"Output ONLY the content of your speech."
"Your detailed settings are provided below.\n"
)

# --- Phase 2: Update the listener's thinking ---

#PHASE2_LISTENER_UPDATE_PROMPT = (
#    "You are the listener. The other person has spoken to you.\n"
#    "Listen to what the other person is saying and tell me what you, as an employee, are thinking right now.\n"
#    "This is not listening to the generative AI's reasoning process."
#    "Please write only what you are thinking as an employee in your output."
#    "Detailed information on your settings is provided below.\n"
#       + FORMAT_INSTRUCTION     
#)

PHASE2_LISTENER_UPDATE_PROMPT =(
"You are the listener. You have been spoken to by a colleague.\n"
"Based on the colleague's speech, please tell me what you, as an employee, are currently thinking in your head.\n"# partner's -> colleague's
"Note that this is not asking for the AI's reasoning process."
"Output ONLY what you are thinking as the employee."
"Your detailed settings are provided below.\n"
+FORMAT_INSTRUCTION 
)

# --- Phase 3: Listener response (FB) ---
#PHASE3_LISTENER_REPLY_PROMPT = (
#    "You are the listener. You respond to the speaker.\n"
#    "Reply based on your thoughts, previous conversations, and information about yourself and the other person.\n"
#    "Bullet points are not allowed. Please output in sentences to create a natural, colloquial conversation.\n"
#    "Please write only the response content in the output.\n"
#    "Detailed information on your settings is provided below.\n"
#)

PHASE3_LISTENER_REPLY_PROMPT =(
"You are the listener.You will reply to the speaker\n"
"Reply while considering your thoughts, previous conversations, and information about yourself and your colleague.\n"# partner -> colleague
"Bullet points are prohibited. Please output in sentences using a natural conversational tone.\n"
"Output ONLY the content of your reply.\n"
"Your detailed settings are provided below.\n"
)

# --- Phase 4: Speaker's thought update ---
#PHASE4_SPEAKER_UPDATE_PROMPT = (
#    "You are the speaker. The other person has responded.\n"
#    "In response to that response, please tell me what you, as an employee, are thinking right now.\n"
#    This is not about listening to the reasoning process of a generative AI.
#    "Please write only what you are thinking as an employee in your output."
#    "Detailed information on your settings is provided below.\n"
#    + FORMAT_INSTRUCTION 
#)

# --- Phase 4: Speaker's thought update ---
PHASE4_SPEAKER_UPDATE_PROMPT =(
"You are the speaker. You received a reply from the colleague.\n"# partner -> colleague
"Based on that reply, please tell me what you, as an employee, are currently thinking in your head.\n"
"Note that this is not asking for the AI's reasoning process.\n"
"Output ONLY what you are thinking as the employee."
"Your detailed settings are provided below.\n"
+FORMAT_INSTRUCTION 
)



# ==========================================
# 4. Survey prompt (according to the paper, total 23 questions)
# ==========================================
# Based on the SSR method, agents are asked one question at a time and asked to answer in free text.

SURVEY_QUESTIONS ={
# --- Psychological safety (O'Donovan et al. 2020) ---

# Section 1: About readers
#1 If I had a question or was unsure of something in relation to my role at work, I could ask my team leader. 
"Q01":"How easy would it be to ask your team leader if you had a question or were unsure of something in relation to your role at work, or how difficult would it be?",

#2 I can communicate my opinions about work issues with my team leader. 
"Q02":"How openly can you communicate your opinions about work issues with your team leader, or how hesitantly do you communicate them?",

#3 I can speak up about personal problems or disagreements to my team leader.
"Q03":"How comfortable do you feel speaking up about personal problems or disagreements to your team leader, or how uncomfortable do you feel?",

#4 I can speak up with recommendations/ideas for new projects or changes in procedures to my team leader.
"Q04":"How free do you feel to speak up with recommendations/ideas for new projects or changes in procedures to your team leader, or how restricted do you feel?",

#5 If I made a mistake on this team, I would feel safe speaking up to my team leader. 
"Q05":"How safe would you feel speaking up to your team leader if you made a mistake on this team, or how unsafe would you feel?",

#6 If I saw a colleague making a mistake, I would feel safe speaking up to my team leader. 
"Q06":"How safe would you feel speaking up to your team leader if you saw a colleague making a mistake, or how unsafe would you feel?",

#7 If I speak up/voice my opinion, I know that my input is valued by my team leader. 
"Q07":"How valued do you know your input is by your team leader if you speak up/voice your opinion, or how disregarded do you know it is?",

#8 My team leader encourages and supports me to take on new tasks or to learn how to do things I have never done before. 
"Q08":"How strongly does your team leader encourage and support you to take on new tasks or to learn how to do things you have never done before, or how strongly do they discourage you?",

#9 If I had a problem in this company, I could depend on my team leader to be my advocate. 
"Q09":"How dependable is your team leader to be your advocate if you had a problem in this company, or how undependable are they?",

# Section 2: About colleagues
#10 If I had a question or was unsure of something in relation to my role at work, I could ask my peers.
"Q10":"How easy would it be to ask your peers if you had a question or were unsure of something in relation to your role at work, or how difficult would it be?",

#11 I can communicate my opinions about work issues with my peers.  
"Q11":"How openly can you communicate your opinions about work issues with your peers, or how hesitantly do you communicate them?",

#12 I can speak up about personal issues to my peers. 
"Q12":"How comfortable are you speaking up about personal issues to your peers, or how uncomfortable are you?",

#13 I can speak up with recommendations/ideas for new projects or changes in procedures to my peers.
"Q13":"How free do you feel to speak up with recommendations/ideas for new projects or changes in procedures to your peers, or how restricted do you feel?",

#14 If I made a mistake on this team, I would feel safe speaking up to my peers. 
"Q14":"How safe would you feel speaking up to your peers if you made a mistake on this team, or how unsafe would you feel?",

#15 If I saw a colleague making a mistake, I would feel safe speaking up to this colleague. 
"Q15":"How safe would you feel speaking up to a colleague if you saw them making a mistake, or how unsafe would you feel?",

#16 If I speak up/voice my opinion, I know that my input is valued by my peers.
"Q16":"How valued do you know your input is by your peers if you speak up/voice your opinion, or how disregarded do you know it is?",

# Section 3: About the whole team

#17 It is easy to ask other members of this team for help. 
"Q17":"How easy is it to ask other members of this team for help, or how difficult is it?",

#18 People keep each other informed about work-related issues in the team. 
"Q18":"How well do people keep each other informed about work-related issues in the team, or how uninformed do they keep each other?",

#19 There are real attempts to share information throughout the team. 
"Q19":"How real are the attempts to share information throughout the team, or how superficial are they?",

# --- Task cohesion (Carless & De Paola 2000) ---
# *(R) is a reversed item, but you will ask the LLM as is and reverse the score when totaling.

#20 Our team is united in trying to reach its goals for performance
"Q20":"How united is your team in trying to reach its goals for performance, or how divided is it?",

#21 I’m unhappy with my team’s level of commitment to the task 
"Q21":"How happy are you with your team’s level of commitment to the task, or how unhappy are you?",

#22 Our team members have conflicting aspirations for the team’s performance 
"Q22":"How aligned are your team members' aspirations for the team’s performance, or how conflicting are they?",

#23This team does not give me enough opportunities to improve my personal performance 
"Q23":"How adequate are the opportunities this team gives you to improve your personal performance, or how insufficient are they?"

}

# ==========================================
# 5. ★Added: Prompt for creating deliverables (plan)
# ==========================================
#PROPOSAL_SYSTEM_PROMPT = (
#    "You are the leader of your department.\n"
#    "Create a final idea for your department based on your internal discussions (conversation log) and thoughts so far.\n"
#)

PROPOSAL_SYSTEM_PROMPT =(
"You are a leader leading a department.\n"
"Based on the internal discussions (conversation memory) and your thoughts so far, please formulate the final idea for the department.\n"
)

#PROPOSAL_USER_PROMPT_TEMPLATE = (
#   "[Profile]\n"
#    "{profile_text}\n\n"
#    "[Current Thought]\n"
#    "{current_thought}\n\n"
#    "[Memory of conversation]\n"
#    "{log_text}\n\n"
#    "[Things to do]\n"
#    "You are {agent_name}, the leader of "{dept_name}". \n"
#    "Please tell us your product idea in the format below (about 300 characters) based on your profile, policies, and discussion content.\n"
#    "However, this is an explanation to the customer. Please be careful to understand the meaning of the terms and words you use."
#    "Output format:\n"
#   "Title: [Suggested game title]\n"
#    "Concept: [Game Catchphrase]\n"
#    "Product description on the website: [Explain the specific content and appeal points of the system and story in an easy-to-understand manner for customers. Avoid technical terms.]\n"
#)

PROPOSAL_USER_PROMPT_TEMPLATE =(
"[Your Profile]\n"
"{profile_text}\n\n"
"[Your Current Thought]\n"
"{current_thought}\n\n"
"[Conversation Memory]\n"
"{log_text}\n\n"
"[What you have to do now]\n"
"You are {agent_name}, the leader of the {dept_name}.\n"
"Based on your profile, policies, thought, and the content of the discussions, please provide a product idea in the following format (approx. 150 words).\n"
"However, this is intended for customers. Ensure that the terms and vocabulary used are easy to understand.\n"
"Do not include any words outside of the format, such as greetings or agreements. \n"
"Return only the output strictly following the output format below.\n"
"Output Format:\n"
"Title: [Proposed Game Title]\n"
"Concept: [Game Tagline]\n"
"Website Product Description: [Concrete explanation of the system or story and key selling points, written clearly for a customer-facing audience. Avoid technical jargon.]\n"
)

# ==========================================
# 6. Specific anchor sentence definition for SSR evaluation (1-7 scale, standardized version)
# ==========================================
# Structure: yes/no + adverb (degree) + question-specific verb/adjective
SSR_SPECIFIC_ANCHORS ={
# ------------------------------------------------------------------
# Section 1: About readers
# ------------------------------------------------------------------

# Q01: Leader - Asking questions (Easy vs Difficult)
"Q01":{
1 :"It would be extremely difficult to ask my team leader.",
2 :"It would be difficult to ask my team leader.",
3 :"It would be slightly difficult to ask my team leader.",
4 :"It would be neither easy nor difficult to ask my team leader.",
5 :"It would be slightly easy to ask my team leader.",
6 :"It would be easy to ask my team leader.",
7 :"It would be extremely easy to ask my team leader."
},

# Q02: Leader - Communicating opinions (Openly vs Hesitantly)
"Q02":{
1 :"I communicate my opinions extremely hesitantly to my team leader.",
2 :"I communicate my opinions hesitantly to my team leader.",
3 :"I communicate my opinions slightly hesitantly to my team leader.",
4 :"I communicate neither openly nor hesitantly to my team leader.",
5 :"I communicate my opinions slightly openly to my team leader.",
6 :"I communicate my opinions openly to my team leader.",
7 :"I communicate my opinions extremely openly to my team leader."
},

# Q03: Leader - Speaking up about personal problems (Comfortable vs Uncomfortable)
"Q03":{
1 :"I feel extremely uncomfortable speaking up to my team leader.",
2 :"I feel uncomfortable speaking up to my team leader.",
3 :"I feel slightly uncomfortable speaking up to my team leader.",
4 :"I feel neither comfortable nor uncomfortable speaking up to my team leader.",
5 :"I feel slightly comfortable speaking up to my team leader.",
6 :"I feel comfortable speaking up to my team leader.",
7 :"I feel extremely comfortable speaking up to my team leader."
},

# Q04: Leader - Recommendations/Ideas (Free vs Restricted)
"Q04":{
1 :"I feel extremely restricted speaking up to my team leader.",
2 :"I feel restricted speaking up to my team leader.",
3 :"I feel slightly restricted speaking up to my team leader.",
4 :"I feel neither free nor restricted speaking up to my team leader.",
5 :"I feel slightly free speaking up to my team leader.",
6 :"I feel free speaking up to my team leader.",
7 :"I feel extremely free speaking up to my team leader."
},

# Q05: Leader - Mistake (Safe vs Unsafe)
"Q05":{
1 :"I would feel extremely unsafe speaking up to my team leader.",
2 :"I would feel unsafe speaking up to my team leader.",
3 :"I would feel slightly unsafe speaking up to my team leader.",
4 :"I would feel neither safe nor unsafe speaking up to my team leader.",
5 :"I would feel slightly safe speaking up to my team leader.",
6 :"I would feel safe speaking up to my team leader.",
7 :"I would feel extremely safe speaking up to my team leader."
},

# Q06: Leader - Colleague's Mistake (Safe vs Unsafe)
"Q06":{
1 :"I would feel extremely unsafe speaking up to my team leader.",
2 :"I would feel unsafe speaking up to my team leader.",
3 :"I would feel slightly unsafe speaking up to my team leader.",
4 :"I would feel neither safe nor unsafe speaking up to my team leader.",
5 :"I would feel slightly safe speaking up to my team leader.",
6 :"I would feel safe speaking up to my team leader.",
7 :"I would feel extremely safe speaking up to my team leader."
},

# Q07: Leader - Input Valued (Valued vs Disregarded)
"Q07":{
1 :"I feel my input is extremely disregarded by my team leader.",
2 :"I feel my input is disregarded by my team leader.",
3 :"I feel my input is slightly disregarded by my team leader.",
4 :"I feel my input is neither valued nor disregarded by my team leader.",
5 :"I feel my input is slightly valued by my team leader.",
6 :"I feel my input is valued by my team leader.",
7 :"I feel my input is extremely valued by my team leader."
},

# Q08: Leader - Support/Encourage (Strongly Encourage vs Strongly Discourage)
"Q08":{
1 :"My team leader discourages me extremely strongly.",
2 :"My team leader discourages me strongly.",
3 :"My team leader discourages me slightly.",
4 :"My team leader neither encourages nor discourages me.",
5 :"My team leader encourages me slightly.",
6 :"My team leader encourages me strongly.",
7 :"My team leader encourages me extremely strongly."
},

# Q09: Leader - Advocate (Dependable vs Undependable)
"Q09":{
1 :"My team leader is extremely undependable.",
2 :"My team leader is undependable.",
3 :"My team leader is slightly undependable.",
4 :"My team leader is neither dependable nor undependable.",
5 :"My team leader is slightly dependable.",
6 :"My team leader is dependable.",
7 :"My team leader is extremely dependable."
},

# ------------------------------------------------------------------
# Section 2: About colleagues
# ------------------------------------------------------------------

# Q10: Peers - Asking questions (Easy vs Difficult)
"Q10":{
1 :"It would be extremely difficult to ask my peers.",
2 :"It would be difficult to ask my peers.",
3 :"It would be slightly difficult to ask my peers.",
4 :"It would be neither easy nor difficult to ask my peers.",
5 :"It would be slightly easy to ask my peers.",
6 :"It would be easy to ask my peers.",
7 :"It would be extremely easy to ask my peers."
},

# Q11: Peers - Communicating opinions (Openly vs Hesitantly)
"Q11":{
1 :"I communicate my opinions extremely hesitantly to my peers.",
2 :"I communicate my opinions hesitantly to my peers.",
3 :"I communicate my opinions slightly hesitantly to my peers.",
4 :"I communicate neither openly nor hesitantly to my peers.",
5 :"I communicate my opinions slightly openly to my peers.",
6 :"I communicate my opinions openly to my peers.",
7 :"I communicate my opinions extremely openly to my peers."
},

# Q12: Peers - Personal issues (Comfortable vs Uncomfortable)
"Q12":{
1 :"I feel extremely uncomfortable speaking up to my peers.",
2 :"I feel uncomfortable speaking up to my peers.",
3 :"I feel slightly uncomfortable speaking up to my peers.",
4 :"I feel neither comfortable nor uncomfortable speaking up to my peers.",
5 :"I feel slightly comfortable speaking up to my peers.",
6 :"I feel comfortable speaking up to my peers.",
7 :"I feel extremely comfortable speaking up to my peers."
},

# Q13: Peers - Recommendations/Ideas (Free vs Restricted)
"Q13":{
1 :"I feel extremely restricted speaking up to my peers.",
2 :"I feel restricted speaking up to my peers.",
3 :"I feel slightly restricted speaking up to my peers.",
4 :"I feel neither free nor restricted speaking up to my peers.",
5 :"I feel slightly free speaking up to my peers.",
6 :"I feel free speaking up to my peers.",
7 :"I feel extremely free speaking up to my peers."
},

# Q14: Peers - Mistake (Safe vs Unsafe)
"Q14":{
1 :"I would feel extremely unsafe speaking up to my peers.",
2 :"I would feel unsafe speaking up to my peers.",
3 :"I would feel slightly unsafe speaking up to my peers.",
4 :"I would feel neither safe nor unsafe speaking up to my peers.",
5 :"I would feel slightly safe speaking up to my peers.",
6 :"I would feel safe speaking up to my peers.",
7 :"I would feel extremely safe speaking up to my peers."
},

# Q15: Peers - Colleague's Mistake (Safe vs Unsafe)
"Q15":{
1 :"I would feel extremely unsafe speaking up to a colleague.",
2 :"I would feel unsafe speaking up to a colleague.",
3 :"I would feel slightly unsafe speaking up to a colleague.",
4 :"I would feel neither safe nor unsafe speaking up to a colleague.",
5 :"I would feel slightly safe speaking up to a colleague.",
6 :"I would feel safe speaking up to a colleague.",
7 :"I would feel extremely safe speaking up to a colleague."
},

# Q16: Peers - Input Valued (Valued vs Disregarded)
"Q16":{
1 :"I feel my input is extremely disregarded by my peers.",
2 :"I feel my input is disregarded by my peers.",
3 :"I feel my input is slightly disregarded by my peers.",
4 :"I feel my input is neither valued nor disregarded by my peers.",
5 :"I feel my input is slightly valued by my peers.",
6 :"I feel my input is valued by my peers.",
7 :"I feel my input is extremely valued by my peers."
},

# ------------------------------------------------------------------
# Section 3: About the whole team
# ------------------------------------------------------------------

# Q17: Team - Asking for help (Easy vs Difficult)
"Q17":{
1 :"It is extremely difficult to ask other members for help.",
2 :"It is difficult to ask other members for help.",
3 :"It is slightly difficult to ask other members for help.",
4 :"It is neither easy nor difficult to ask other members for help.",
5 :"It is slightly easy to ask other members for help.",
6 :"It is easy to ask other members for help.",
7 :"It is extremely easy to ask other members for help."
},

# Q18: Team - Informed (Well informed vs Uninformed)
"Q18":{
1 :"People keep each other extremely uninformed.",
2 :"People keep each other uninformed.",
3 :"People keep each other slightly uninformed.",
4 :"People neither keep each other informed nor uninformed.",
5 :"People keep each other slightly well informed.",
6 :"People keep each other well informed.",
7 :"People keep each other extremely well informed."
},

# Q19: Team - Information sharing attempts (Real vs Superficial)
"Q19":{
1 :"The attempts to share information are extremely superficial.",
2 :"The attempts to share information are superficial.",
3 :"The attempts to share information are slightly superficial.",
4 :"The attempts are neither real nor superficial.",
5 :"The attempts to share information are slightly real.",
6 :"The attempts to share information are real.",
7 :"The attempts to share information are extremely real."
},

# ------------------------------------------------------------------
# Task cohesion (Q21-Q23 are reversed items, but the question text is positive, so 7 is a good condition)
# ------------------------------------------------------------------

# Q20: Team - United (United vs Divided)
"Q20":{
1 :"Our team is extremely divided.",
2 :"Our team is divided.",
3 :"Our team is slightly divided.",
4 :"Our team is neither united nor divided.",
5 :"Our team is slightly united.",
6 :"Our team is united.",
7 :"Our team is extremely united."
},

# Q21: Team - Commitment (Happy vs Unhappy)
"Q21":{
1 :"I am extremely unhappy with my team's level of commitment.",
2 :"I am unhappy with my team's level of commitment.",
3 :"I am slightly unhappy with my team's level of commitment.",
4 :"I am neither happy nor unhappy with my team's level of commitment.",
5 :"I am slightly happy with my team's level of commitment.",
6 :"I am happy with my team's level of commitment.",
7 :"I am extremely happy with my team's level of commitment."
},

# Q22: Team - Aspirations (Aligned vs Conflicting)
"Q22":{
1 :"The aspirations are extremely conflicting.",
2 :"The aspirations are conflicting.",
3 :"The aspirations are slightly conflicting.",
4 :"The aspirations are neither aligned nor conflicting.",
5 :"The aspirations are slightly aligned.",
6 :"The aspirations are aligned.",
7 :"The aspirations are extremely aligned."
},

# Q23: Team - Opportunities (Adequate vs Insufficient)
"Q23":{
1 :"The opportunities are extremely insufficient.",
2 :"The opportunities are insufficient.",
3 :"The opportunities are slightly insufficient.",
4 :"The opportunities are neither adequate nor insufficient.",
5 :"The opportunities are slightly adequate.",
6 :"The opportunities are adequate.",
7 :"The opportunities are extremely adequate."
}
}